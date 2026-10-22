#ضمن "ggml-vulkan-common.h"

#ضمن <cstdint>
#ضمن <cstdlib>
#ضمن <mutex>
#ضمن <shared_mutex>

ساكن فراغ * portable_aligned_malloc(size_t align, size_t size) {
#اگر_تعریف_شده _WIN32
    أرجع _aligned_malloc(size, align);
#وإلا
    فراغ * ptr = اشاره_تهی;
    إذا (posix_memalign(&ptr, align, size) != 0) {
        أرجع اشاره_تهی;
    }
    أرجع ptr;
#پایان_اگر
}

ساكن فراغ portable_aligned_free(فراغ * ptr) {
#اگر_تعریف_شده _WIN32
    _aligned_free(ptr);
#وإلا
    std::حرر(ptr);
#پایان_اگر
}

ggml_backend_میانگیر_type_i ggml_backend_vk_میانگیر_type_interface = {
    /* .get_name            = */ ggml_backend_vk_میانگیر_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_میانگیر_type_alloc_میانگیر,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_vk_میانگیر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_میانگیر_type_get_max_اندازه,
    /* .get_alloc_size      = */ ggml_backend_vk_میانگیر_type_get_alloc_اندازه,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ عدم,
};

ساكن std::vector<uint32_t> ggml_vk_پیدا_کن_حافظه_properties(ثابت vk::Physical_Device_حافظه_Properties* mem_props, vk::حافظه_Requirements* mem_req, vk::حافظه_Property_Flags flags) {
    std::vector<uint32_t> indices;

    لكل (uint32_t i = 0; i < mem_props->حافظه_Type_شمار; ++i) {
        vk::حافظه_Type حافظه_type = mem_props->حافظه_Types[i];
        إذا ((mem_req->حافظه_Type_بیتها & ((uint64_t)1 << i)) &&
            (flags & حافظه_type.propertyFlags) == flags &&
            mem_props->حافظه_Heaps[حافظه_type.انباشته_شاخص].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    أرجع indices;
}

ساكن vk_میانگیر ggml_vk_create_میانگیر(vk_device& device, size_t size, ثابت std::initializer_list<vk::حافظه_Property_Flags> & req_flags_list,
                                       فراغ *import_ptr = اشاره_تهی) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    إذا (size > device->max_میانگیر_اندازه) {
        ارم vk::بیرون_از_Device_حافظه_Error("Requested buffer size exceeds device buffer size limit");
    }

    vk_میانگیر buf = std::make_shared<vk_میانگیر_struct>();

    إذا (size == 0) {
        buf->size = 0;
        أرجع buf;
    }

    vk::میانگیر_Usage_Flags usage_flags = vk::میانگیر_Usage_Flag_بیتها::e_Storage_میانگیر | vk::میانگیر_Usage_Flag_بیتها::eTransferSrc | vk::میانگیر_Usage_Flag_بیتها::eTransferDst;
    vk::حافظه_Allocate_Flags mem_flags {};
    إذا (device->میانگیر_device_آدرس) {
        usage_flags |= vk::میانگیر_Usage_Flag_بیتها::e_Shader_Device_آدرس;
        mem_flags |= vk::حافظه_Allocate_Flag_بیتها::e_Device_آدرس;
    }

    vk::میانگیر_Create_Info میانگیر_create_info{
        vk::میانگیر_Create_Flags(),
        size,
        usage_flags,
        vk::Sharing_نما::eExclusive,
        0,
        اشاره_تهی,
    };

    vk::External_حافظه_میانگیر_Create_Info external_حافظه_bci;
    إذا (import_ptr) {
        external_حافظه_bci.handleTypes = vk::External_حافظه_Handle_Type_Flag_بیتها::e_Host_تخصیص_EXT;
        میانگیر_create_info.مجموعه_PNext(&external_حافظه_bci);
    }

    buf->میانگیر = device->device.create_میانگیر(میانگیر_create_info);

    vk::حافظه_Requirements mem_req = device->device.get_میانگیر_حافظه_Requirements(buf->میانگیر);

    vk::Physical_Device_حافظه_Properties mem_props = device->physical_device.get_حافظه_Properties();

    ثابت vk::حافظه_اولویت_Allocate_Info_EXT mem_اولویت_info { 1.0f };

    vk::حافظه_Allocate_Flags_Info mem_flags_info { mem_flags };

    إذا (device->حافظه_اولویت) {
        mem_flags_info.مجموعه_PNext(&mem_اولویت_info);
    }

    إذا (import_ptr) {
        vk::حافظه_Host_اشاره_گر_Properties_EXT host_اشاره_گر_props;
        حاول {
            host_اشاره_گر_props = device->device.get_حافظه_Host_اشاره_گر_Properties_EXT(vk::External_حافظه_Handle_Type_Flag_بیتها::e_Host_تخصیص_EXT, import_ptr);
        } امسك (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroy_میانگیر(buf->میانگیر);
            أرجع {};
        }
        vk::Physical_Device_حافظه_Properties mem_props = device->physical_device.get_حافظه_Properties();

        uint32_t حافظه_type_idx;
        vk::حافظه_Property_Flags property_flags = *req_flags_list.begin();
        لكل (حافظه_type_idx = 0; حافظه_type_idx < 32; ++حافظه_type_idx) {
            إذا (!(host_اشاره_گر_props.حافظه_Type_بیتها & (1u << حافظه_type_idx))) {
                استمر;
            }
            إذا (!(mem_req.حافظه_Type_بیتها & (1u << حافظه_type_idx))) {
                استمر;
            }

            vk::حافظه_Type حافظه_type = mem_props.حافظه_Types[حافظه_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            إذا ((حافظه_type.propertyFlags & property_flags) == property_flags) {
                property_flags = حافظه_type.propertyFlags;
                اكسر;
            }
        }
        إذا (حافظه_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroy_میانگیر(buf->میانگیر);
            أرجع {};
        }

        buf->حافظه_property_flags = mem_props.حافظه_Types[حافظه_type_idx].propertyFlags;
        حاول {
            vk::Import_حافظه_Host_اشاره_گر_Info_EXT import_info;
            import_info.handleType = vk::External_حافظه_Handle_Type_Flag_بیتها::e_Host_تخصیص_EXT;
            import_info.p_Host_اشاره_گر = import_ptr;
            import_info.مجموعه_PNext(&mem_flags_info);
            buf->device_حافظه = device->device.allocate_حافظه({ size, حافظه_type_idx, &import_info });
        } امسك (ثابت vk::SystemError& e) {
        }
    } وإلا {
        لكل (خودکار it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            ثابت خودکار & req_flags = *it;

            ثابت std::vector<uint32_t> حافظه_type_indices = ggml_vk_پیدا_کن_حافظه_properties(&mem_props, &mem_req, req_flags);

            إذا (حافظه_type_indices.خالی()) {
                استمر;
            }

            منطقي انجامشده = خطأ;

            لكل (خودکار mtype_it = حافظه_type_indices.begin(); mtype_it != حافظه_type_indices.end(); mtype_it++) {
                حاول {
                    buf->device_حافظه = device->device.allocate_حافظه({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->حافظه_property_flags = mem_props.حافظه_Types[*mtype_it].propertyFlags;
                    انجامشده = درست;
                    اكسر;
                } امسك (ثابت vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    إذا (it + 1 == req_flags_list.end() && mtype_it + 1 == حافظه_type_indices.end()) {
                        device->device.destroy_میانگیر(buf->میانگیر);
                        ارم e;
                    }
                }
            }

            إذا (انجامشده) {
                اكسر;
            }
        }
    }

    إذا (!buf->device_حافظه) {
        device->device.destroy_میانگیر(buf->میانگیر);
        ارم vk::بیرون_از_Device_حافظه_Error("No suitable memory type found");
    }

    buf->ptr = اشاره_تهی;

    إذا (import_ptr) {
        buf->ptr = import_ptr;
    } وإلا {
        إذا (buf->حافظه_property_flags & vk::حافظه_Property_Flag_بیتها::eHostVisible) {
            buf->ptr = device->device.map_حافظه(buf->device_حافظه, 0, VK_کامل_اندازه);
        }
    }

    device->device.bind_میانگیر_حافظه(buf->میانگیر, buf->device_حافظه, 0);

    buf->device = device;
    buf->size = size;

    إذا (device->میانگیر_device_آدرس) {
        ثابت vk::میانگیر_Device_آدرس_Info آدرس_Info(buf->میانگیر);
        buf->bda_addr = device->device.get_میانگیر_آدرس(آدرس_Info);
    }

    device->حافظه_logger->log_تخصیص(buf, size);

    أرجع buf;
}

vk_میانگیر ggml_vk_create_میانگیر_check(vk_device& device, size_t size, vk::حافظه_Property_Flags req_flags, vk::حافظه_Property_Flags fallback_flags) {
    حاول {
        أرجع ggml_vk_create_میانگیر(device, size, {req_flags, fallback_flags});
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::پایان_سطر;
        std::cerr << "ggml_vulkan: " << e.what() << std::پایان_سطر;
        ارم e;
    }
}

vk_میانگیر ggml_vk_create_میانگیر_device(vk_device& device, size_t size) {
    vk_میانگیر buf;
    حاول {
        إذا (device->prefer_host_حافظه) {
            buf = ggml_vk_create_میانگیر(device, size, {vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent,
                                                       vk::حافظه_Property_Flag_بیتها::eDeviceLocal});
        } وإلا إذا (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_میانگیر(device, size, {vk::حافظه_Property_Flag_بیتها::eDeviceLocal | vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent,
                                                       vk::حافظه_Property_Flag_بیتها::eDeviceLocal,
                                                       vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent});
        } وإلا إذا (device->disable_host_visible_vidmem) {
            إذا (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_میانگیر(device, size, {vk::حافظه_Property_Flag_بیتها::eDeviceLocal,
                                                           vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent});
            } وإلا {
                buf = ggml_vk_create_میانگیر(device, size, {vk::حافظه_Property_Flag_بیتها::eDeviceLocal});
            }
        } وإلا {
            // use rebar if available, otherwise fallback to device only visible memory
            إذا (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_میانگیر(device, size, {vk::حافظه_Property_Flag_بیتها::eDeviceLocal | vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent,
                                                           vk::حافظه_Property_Flag_بیتها::eDeviceLocal,
                                                           vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent});
            } وإلا {
                buf = ggml_vk_create_میانگیر(device, size, {vk::حافظه_Property_Flag_بیتها::eDeviceLocal | vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent,
                                                           vk::حافظه_Property_Flag_بیتها::eDeviceLocal});
            }
        }
    } امسك (ثابت vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::پایان_سطر;
        std::cerr << "ggml_vulkan: " << e.what() << std::پایان_سطر;
        ارم e;
    }

    أرجع buf;
}

فراغ ggml_vk_destroy_میانگیر(vk_میانگیر& buf) {
    إذا (buf == اشاره_تهی) {
        أرجع;
    }

    إذا (buf->device != اشاره_تهی) {
        buf->device->حافظه_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
ساكن size_t ggml_vk_host_تخصیص_پیدا_کن(ثابت فراغ * ptr) {
    لكل (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        ثابت uint8_t * پایه = (ثابت uint8_t *) vk_instance.host_allocations[i].پایه;
        إذا (ptr >= پایه && ptr < پایه + vk_instance.host_allocations[i].size) {
            أرجع i;
        }
    }
    أرجع SIZE_MAX;
}

فراغ * ggml_vk_host_malloc(size_t size) {
    VK_LOG_حافظه("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    إذا (vk_instance.multi_device) {
        ثابت size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        ثابت size_t alloc_اندازه = (size + align - 1) / align * align;

        فراغ * ptr = portable_aligned_malloc(align, alloc_اندازه);
        إذا (ptr == اشاره_تهی) {
            أرجع اشاره_تهی;
        }
        vk_host_تخصیص مدخل;
        مدخل.پایه = ptr;
        مدخل.size = alloc_اندازه;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(مدخل));
        أرجع ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_میانگیر buf;
    حاول {
        buf = ggml_vk_create_میانگیر(dev0, size,
            {vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent | vk::حافظه_Property_Flag_بیتها::eHostCached,
             vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent});
    } امسك (vk::SystemError&) {
        أرجع اشاره_تهی;
    }

    إذا (!(buf->حافظه_property_flags & vk::حافظه_Property_Flag_بیتها::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_میانگیر(buf);
        أرجع اشاره_تهی;
    }

    vk_host_تخصیص مدخل;
    مدخل.پایه             = buf->ptr;
    مدخل.size             = size;
    مدخل.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(مدخل));
    أرجع buf->ptr;
}

فراغ ggml_vk_host_free(فراغ * ptr) {
    إذا (ptr == اشاره_تهی) {
        أرجع;
    }
    VK_LOG_حافظه("ggml_vk_host_free(" << ptr << ")");

    vk_host_تخصیص مدخل;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        ثابت size_t شاخص = ggml_vk_host_تخصیص_پیدا_کن(ptr);
        إذا (شاخص == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            أرجع;
        }
        مدخل = std::move(vk_instance.host_allocations[شاخص]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + شاخص);
    }

    لكل (خودکار & view : مدخل.views) {
        ggml_vk_destroy_میانگیر(view);
    }
    إذا (vk_instance.multi_device) {
        portable_aligned_free(مدخل.پایه);
    }
}

فراغ ggml_vk_host_get(ثابت vk_device& device, ثابت فراغ * ptr, vk_میانگیر& buf, size_t& buf_جابه_جایی) {
    buf = اشاره_تهی;
    buf_جابه_جایی = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    ثابت size_t شاخص = ggml_vk_host_تخصیص_پیدا_کن(ptr);
    إذا (شاخص == SIZE_MAX) {
        أرجع;
    }
    vk_host_تخصیص & مدخل = vk_instance.host_allocations[شاخص];

    إذا (مدخل.views[device->idx] == اشاره_تهی) {
        ثابت uint32_t بیت = 1u << device->idx;
        إذا (!vk_instance.multi_device || !vk_instance.device_host_میانگیر_import[device->idx] || (vk_instance.host_import_failed & بیت)) {
            أرجع;
        }
        GGML_ASSERT(device->external_حافظه_host);
        // const only propagates from the const context; the pointee is mutable
        مدخل.views[device->idx] = ggml_vk_میانگیر_from_host_ptr(const_cast<vk_device&>(device), مدخل.پایه, مدخل.size);
        إذا (مدخل.views[device->idx] == اشاره_تهی) {
            vk_instance.host_import_failed |= بیت;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            أرجع;
        }
    }

    buf = مدخل.views[device->idx];
    buf_جابه_جایی = (ثابت uint8_t *) ptr - (ثابت uint8_t *) مدخل.پایه;
}

فراغ ggml_vk_ensure_sync_staging_میانگیر(vk_device& device, size_t size) {
    إذا (device->sync_staging == اشاره_تهی || device->sync_staging->size < size) {
        VK_LOG_حافظه("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_میانگیر(device->sync_staging);
        device->sync_staging = ggml_vk_create_میانگیر_check(device, size,
            vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent | vk::حافظه_Property_Flag_بیتها::eHostCached,
            vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent);
    }
}

فراغ ggml_vk_ensure_sync_staging_میانگیر(ggml_backend_vk_context * ctx, size_t size) {
    إذا (ctx->sync_staging == اشاره_تهی || ctx->sync_staging->size < size) {
        VK_LOG_حافظه("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_میانگیر(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_میانگیر_check(ctx->device, size,
            vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent | vk::حافظه_Property_Flag_بیتها::eHostCached,
            vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent);
    }
}

ساكن فراغ ggml_vk_میانگیر_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_میانگیر& dst, size_t جابه_جایی, ثابت ggml_tensor * tensor, منطقي sync_staging = خطأ) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    إذا(dst->حافظه_property_flags & vk::حافظه_Property_Flag_بیتها::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::پایان_سطر;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_میانگیر buf = اشاره_تهی;
    size_t buf_جابه_جایی = 0;
    ggml_vk_host_get(ctx->device, tensor->داده, buf, buf_جابه_جایی);

    ثابت uint64_t ne0 = tensor->ne[0];
    ثابت uint64_t ne1 = tensor->ne[1];
    ثابت uint64_t ne2 = tensor->ne[2];
    ثابت uint64_t ne3 = tensor->ne[3];
    ثابت uint64_t nb0 = tensor->nb[0];
    ثابت uint64_t nb1 = tensor->nb[1];
    ثابت uint64_t nb2 = tensor->nb[2];
    ثابت uint64_t nb3 = tensor->nb[3];
    ثابت ggml_type type = tensor->type;
    ثابت uint64_t ts = ggml_type_اندازه(type);
    ثابت uint64_t bs = ggml_blck_اندازه(type);

    ثابت uint64_t dstnb0 = ts;
    ثابت uint64_t dstnb1 = dstnb0*(ne0/bs);
    ثابت uint64_t dstnb2 = dstnb1*ne1;
    ثابت uint64_t dstnb3 = dstnb2*ne2;

    ثابت uint64_t ne = ggml_nelements(tensor);

    إذا (buf != اشاره_تهی) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::میانگیر_Copy> slices;

        لكل (uint64_t i3 = 0; i3 < ne3; i3++) {
            لكل (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                إذا (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_جابه_جایی + i3*nb3 + i2*nb2, جابه_جایی + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } وإلا {
                    لكل (uint64_t i1 = 0; i1 < ne1; i1++) {
                        إذا (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_جابه_جایی + i3*nb3 + i2*nb2 + i1*nb1, جابه_جایی + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } وإلا {
                            ثابت uint64_t s_off = buf_جابه_جایی + i3*nb3 + i2*nb2 + i1*nb1;
                            ثابت uint64_t d_off = جابه_جایی + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            لكل (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->میانگیر->buf.copy_میانگیر(buf->میانگیر, dst->میانگیر, slices);
        أرجع;
    }

    إذا (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_میانگیر& staging = ctx->device->sync_staging;
    ثابت uint64_t copy_اندازه = ts*ne/bs;
    ggml_vk_ensure_sync_staging_میانگیر(ctx->device, copy_اندازه);
    Vk_میانگیر_Copy buf_copy{ 0, جابه_جایی, copy_اندازه };

    ggml_vk_sync_buffers(ctx, subctx);
    vk_Cmd_Copy_میانگیر(subctx->s->میانگیر->buf, (Vk_میانگیر)staging->میانگیر, (Vk_میانگیر)dst->میانگیر, 1, &buf_copy);

    لكل (uint64_t i3 = 0; i3 < ne3; i3++) {
        لكل (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            إذا (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (ثابت uint8_t *) tensor->داده + buf_جابه_جایی + i3*nb3 + i2*nb2, dstnb2, &subctx->درون_memcpys);
            } وإلا {
                لكل (uint64_t i1 = 0; i1 < ne1; i1++) {
                    إذا (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (ثابت uint8_t *) tensor->داده + buf_جابه_جایی + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->درون_memcpys);
                    } وإلا {
                        ثابت uint64_t s_off = buf_جابه_جایی + i3*nb3 + i2*nb2 + i1*nb1;
                        ثابت uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        لكل (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (ثابت uint8_t *) tensor->داده + s_off + i0*nb0, dstnb0, &subctx->درون_memcpys);
                        }
                    }
                }
            }
        }
    }
}

منطقي ggml_vk_میانگیر_write_2d_async(vk_context subctx, vk_میانگیر& dst, size_t جابه_جایی, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t عرض, size_t ارتفاع, منطقي sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << عرض << ", " << ارتفاع << ")");
    // Check if src is pinned memory
    vk_میانگیر buf = اشاره_تهی;
    size_t buf_جابه_جایی = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_جابه_جایی);

    إذا (buf != اشاره_تهی) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::میانگیر_Copy> slices(1);
        إذا (عرض == spitch && عرض == dpitch) {
            // Only do single write if stride is equal
            slices[0].src_جابه_جایی_2 = buf_جابه_جایی;
            slices[0].dst_جابه_جایی_2 = جابه_جایی;
            slices[0].size = عرض * ارتفاع;
        } وإلا {
            slices.resize(ارتفاع);
            لكل (size_t i = 0; i < ارتفاع; i++) {
                slices[i].src_جابه_جایی_2 = buf_جابه_جایی + i * spitch;
                slices[i].dst_جابه_جایی_2 = جابه_جایی + i * dpitch;
                slices[i].size = عرض;
            }
        }

        ggml_vk_sync_buffers(اشاره_تهی, subctx);
        subctx->s->میانگیر->buf.copy_میانگیر(buf->میانگیر, dst->میانگیر, slices);
        أرجع درست;
    }
    VK_LOG_DEBUG("STAGING");

    إذا (!sync_staging) {
        // copy was not handled caller needs to fall back
        أرجع خطأ;
    }

    // Staging buffer required
    ثابت size_t staging_اندازه = عرض * ارتفاع;
    ggml_vk_ensure_sync_staging_میانگیر(dst->device, staging_اندازه);

    vk_میانگیر& staging_میانگیر = dst->device->sync_staging;

    std::vector<vk::میانگیر_Copy> slices(1);
    إذا (عرض == dpitch) {
        slices[0].src_جابه_جایی_2 = 0;
        slices[0].dst_جابه_جایی_2 = جابه_جایی;
        slices[0].size = staging_اندازه;
    } وإلا {
        slices.resize(ارتفاع);
        لكل (size_t i = 0; i < ارتفاع; i++) {
            slices[i].src_جابه_جایی_2 = i * عرض;
            slices[i].dst_جابه_جایی_2 = جابه_جایی + i * dpitch;
            slices[i].size = عرض;
        }
    }

    ggml_vk_sync_buffers(اشاره_تهی, subctx);
    subctx->s->میانگیر->buf.copy_میانگیر(staging_میانگیر->میانگیر, dst->میانگیر, slices);

    إذا (عرض == spitch) {
        deferred_memcpy((uint8_t *)staging_میانگیر->ptr, src, staging_اندازه, &subctx->درون_memcpys);
    } وإلا {
        لكل (size_t i = 0; i < ارتفاع; i++) {
            deferred_memcpy((uint8_t *)staging_میانگیر->ptr + i * عرض, (ثابت uint8_t *) src + i * spitch, عرض, &subctx->درون_memcpys);
        }
    }
    أرجع درست;
}

منطقي ggml_vk_میانگیر_write_async(vk_context subctx, vk_میانگیر& dst, size_t جابه_جایی, ثابت فراغ * src, size_t size, منطقي sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    أرجع ggml_vk_میانگیر_write_2d_async(subctx, dst, جابه_جایی, src, size, size, size, 1, sync_staging);
}

فراغ ggml_vk_میانگیر_write_2d(vk_میانگیر& dst, size_t جابه_جایی, ثابت فراغ * src, size_t spitch, size_t dpitch, size_t عرض, size_t ارتفاع) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << عرض << ", " << ارتفاع << ")");
    // Buffer is already mapped
    إذا(dst->حافظه_property_flags & vk::حافظه_Property_Flag_بیتها::eHostVisible) {
        GGML_ASSERT(dst->حافظه_property_flags & vk::حافظه_Property_Flag_بیتها::eHostCoherent);

        إذا (عرض == spitch && عرض == dpitch) {
            کپی_حافظه((uint8_t *)dst->ptr + جابه_جایی, src, عرض * ارتفاع);
        } وإلا {
            لكل (size_t i = 0; i < ارتفاع; i++) {
                کپی_حافظه((uint8_t *)dst->ptr + جابه_جایی + i * dpitch, (ثابت uint8_t *) src + i * spitch, عرض);
            }
        }
    } وإلا {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_صف->cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        منطقي ret = ggml_vk_میانگیر_write_2d_async(subctx, dst, جابه_جایی, src, spitch, dpitch, عرض, ارتفاع, درست);
        GGML_ASSERT(ret);
        ggml_vk_ctx_انتها(subctx);

        لكل (خودکار& cpy : subctx->درون_memcpys) {
            کپی_حافظه(cpy.dst, cpy.src, cpy.n);
        }

        لكل (خودکار& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, درست, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
        ggml_vk_صف_command_pools_cleanup(dst->device);
    }
}

فراغ ggml_vk_میانگیر_write(vk_میانگیر& dst, size_t جابه_جایی, ثابت فراغ * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_میانگیر_write_2d(dst, جابه_جایی, src, size, size, size, 1);
}

منطقي ggml_vk_میانگیر_read_2d_async(vk_context subctx, vk_میانگیر& src, size_t جابه_جایی, فراغ * dst, size_t spitch, size_t dpitch, size_t عرض, size_t ارتفاع, منطقي sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << جابه_جایی << ", width=" << عرض << ", height=" << ارتفاع << ")");
    GGML_ASSERT(عرض > 0);
    GGML_ASSERT(ارتفاع > 0);
    GGML_ASSERT(src != اشاره_تهی);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_میانگیر buf = اشاره_تهی;
    size_t buf_جابه_جایی = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_جابه_جایی);

    std::vector<vk::میانگیر_Copy> slices(1);
    إذا (عرض == spitch && عرض == dpitch) {
        // Only do single write if stride is equal
        slices[0].src_جابه_جایی_2 = جابه_جایی;
        slices[0].dst_جابه_جایی_2 = buf_جابه_جایی;
        slices[0].size = عرض * ارتفاع;
    } وإلا {
        slices.resize(ارتفاع);
        لكل (size_t i = 0; i < ارتفاع; i++) {
            slices[i].src_جابه_جایی_2 = جابه_جایی + i * spitch;
            slices[i].dst_جابه_جایی_2 = buf_جابه_جایی + i * dpitch;
            slices[i].size = عرض;
        }
    }

    إذا (buf != اشاره_تهی) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(اشاره_تهی, subctx);
        subctx->s->میانگیر->buf.copy_میانگیر(src->میانگیر, buf->میانگیر, slices);

        أرجع درست;
    }
    VK_LOG_DEBUG("STAGING");

    إذا (!sync_staging) {
        // copy was not handled caller needs to fall back
        أرجع خطأ;
    }

    // Fall back to staging buffer
    ثابت size_t staging_اندازه = عرض * ارتفاع;
    ggml_vk_ensure_sync_staging_میانگیر(src->device, staging_اندازه);

    vk_میانگیر& staging_میانگیر = src->device->sync_staging;

    std::vector<vk::میانگیر_Copy> staging_slices(1);
    إذا (عرض == spitch) {
        staging_slices[0].src_جابه_جایی_2 = جابه_جایی;
        staging_slices[0].dst_جابه_جایی_2 = 0;
        staging_slices[0].size = staging_اندازه;
    } وإلا {
        staging_slices.resize(ارتفاع);
        لكل (size_t i = 0; i < ارتفاع; i++) {
            staging_slices[i].src_جابه_جایی_2 = جابه_جایی + i * spitch;
            staging_slices[i].dst_جابه_جایی_2 = i * عرض;
            staging_slices[i].size = عرض;
        }
    }

    ggml_vk_sync_buffers(اشاره_تهی, subctx);
    subctx->s->میانگیر->buf.copy_میانگیر(src->میانگیر, staging_میانگیر->میانگیر, staging_slices);

    إذا (عرض == dpitch) {
        deferred_memcpy(dst, staging_میانگیر->ptr, staging_اندازه, &subctx->بیرون_memcpys);
    } وإلا {
        لكل (size_t i = 0; i < ارتفاع; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (ثابت uint8_t *) staging_میانگیر->ptr + i * عرض, عرض, &subctx->بیرون_memcpys);
        }
    }
    أرجع درست;
}

ساكن منطقي ggml_vk_میانگیر_read_async(vk_context subctx, vk_میانگیر& src, size_t جابه_جایی, فراغ * dst, size_t size, منطقي sync_staging = خطأ) {
    أرجع ggml_vk_میانگیر_read_2d_async(subctx, src, جابه_جایی, dst, size, size, size, 1, sync_staging);
}

فراغ ggml_vk_میانگیر_read_2d(vk_میانگیر& src, size_t جابه_جایی, فراغ * dst, size_t spitch, size_t dpitch, size_t عرض, size_t ارتفاع) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->میانگیر << ", " << جابه_جایی << ", " << عرض << ", " << ارتفاع << ")");

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    ثابت منطقي slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->حافظه_property_flags & vk::حافظه_Property_Flag_بیتها::eHostCached);
    إذا(src->حافظه_property_flags & vk::حافظه_Property_Flag_بیتها::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->حافظه_property_flags & vk::حافظه_Property_Flag_بیتها::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_صف->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->میانگیر->buf.pipelineBarrier(
            vk::Pipeline_Stage_Flag_بیتها::eComputeShader | vk::Pipeline_Stage_Flag_بیتها::eTransfer,
            vk::Pipeline_Stage_Flag_بیتها::eHost,
            {},
            { { vk::Access_Flag_بیتها::eShaderWrite | vk::Access_Flag_بیتها::eTransferWrite,
                vk::Access_Flag_بیتها::eHostRead } },
            {}, {});
        ggml_vk_ctx_انتها(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, درست, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_صف_command_pools_cleanup(src->device);

        إذا (عرض == spitch && عرض == dpitch) {
            کپی_حافظه(dst, (ثابت uint8_t *) src->ptr + جابه_جایی, عرض * ارتفاع);
        } وإلا {
            لكل (size_t i = 0; i < ارتفاع; i++) {
                کپی_حافظه((uint8_t *) dst + i * dpitch, (ثابت uint8_t *) src->ptr + جابه_جایی + i * spitch, عرض);
            }
        }
    } وإلا {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_صف->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        منطقي ret = ggml_vk_میانگیر_read_2d_async(subctx, src, جابه_جایی, dst, spitch, dpitch, عرض, ارتفاع, درست);
        GGML_ASSERT(ret);
        ggml_vk_ctx_انتها(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, درست, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_صف_command_pools_cleanup(src->device);

        لكل (خودکار& cpy : subctx->بیرون_memcpys) {
            کپی_حافظه(cpy.dst, cpy.src, cpy.n);
        }
    }
}

فراغ ggml_vk_میانگیر_read(vk_میانگیر& src, size_t جابه_جایی, فراغ * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->میانگیر << ", " << جابه_جایی << ", " << size << ")");
    ggml_vk_میانگیر_read_2d(src, جابه_جایی, dst, size, size, size, 1);
}

فراغ ggml_vk_میانگیر_copy_async(vk_context& ctx, vk_میانگیر& dst, size_t dst_جابه_جایی, vk_میانگیر& src, size_t src_جابه_جایی, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    Vk_میانگیر_Copy bc{ src_جابه_جایی, dst_جابه_جایی, size };

    vk_Cmd_Copy_میانگیر(ctx->s->میانگیر->buf, (Vk_میانگیر)src->میانگیر, (Vk_میانگیر)dst->میانگیر, 1, &bc);
}

فراغ ggml_vk_میانگیر_copy(vk_میانگیر& dst, size_t dst_جابه_جایی, vk_میانگیر& src, size_t src_جابه_جایی, size_t size) {
    إذا (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_صف->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_میانگیر_copy_async(subctx, dst, dst_جابه_جایی, src, src_جابه_جایی, size);
        ggml_vk_ctx_انتها(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, درست, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_صف_command_pools_cleanup(src->device);
    } وإلا {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_میانگیر(src->device, size);

        // Copy to src staging buffer
        ggml_vk_میانگیر_copy(src->device->sync_staging, 0, src, src_جابه_جایی, size);
        // Copy to dst buffer
        ggml_vk_میانگیر_write(dst, dst_جابه_جایی, src->device->sync_staging->ptr, size);
    }
}

فراغ ggml_vk_میانگیر_memset_async(vk_context& ctx, vk_میانگیر& dst, size_t جابه_جایی, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << جابه_جایی << ", " << c << ", " << size << ")");

    إذا (dst->حافظه_property_flags & vk::حافظه_Property_Flag_بیتها::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + جابه_جایی, c, size, &ctx->memsets);
        أرجع;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->میانگیر->buf.fill_میانگیر(dst->میانگیر, جابه_جایی, size, c);
}

فراغ ggml_vk_میانگیر_memset(vk_میانگیر& dst, size_t جابه_جایی, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << جابه_جایی << ", " << c << ", " << size << ")");

    إذا (dst->حافظه_property_flags & vk::حافظه_Property_Flag_بیتها::eHostVisible &&
        dst->device->uma) {
        memset((uint8_t*)dst->ptr + جابه_جایی, c, size);
        أرجع;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_صف->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->میانگیر->buf.fill_میانگیر(dst->میانگیر, جابه_جایی, size, c);
    ggml_vk_ctx_انتها(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, درست, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_صف_command_pools_cleanup(dst->device);
}

ggml_backend_میانگیر_i ggml_backend_vk_میانگیر_interface = {
    /* .free_buffer     = */ ggml_backend_vk_میانگیر_free_میانگیر,
    /* .get_base        = */ ggml_backend_vk_میانگیر_get_پایه,
    /* .init_tensor     = */ ggml_backend_vk_میانگیر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_میانگیر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_میانگیر_مجموعه_tensor,
    /* .get_tensor      = */ ggml_backend_vk_میانگیر_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_میانگیر_مجموعه_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_میانگیر_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_میانگیر_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_میانگیر_clear,
    /* .reset           = */ عدم,
};

vk_میانگیر ggml_vk_میانگیر_from_host_ptr(vk_device & device, فراغ * ptr, size_t size) {
    إذا (!device->external_حافظه_host) {
        أرجع {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    إذا (uptr & (device->min_imported_host_اشاره_گر_alignment - 1)) {
        أرجع {};
    }
    إذا (size & (device->min_imported_host_اشاره_گر_alignment - 1)) {
        أرجع {};
    }

    ثابت vk::حافظه_Property_Flags property_flags = vk::حافظه_Property_Flag_بیتها::eHostVisible | vk::حافظه_Property_Flag_بیتها::eHostCoherent | vk::حافظه_Property_Flag_بیتها::eHostCached;

    vk_میانگیر buf {};
    حاول {
        buf = ggml_vk_create_میانگیر(device, size, { property_flags }, ptr);
    } امسك (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    أرجع buf;
}

