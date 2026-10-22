#ضمن <filesystem>
#ضمن <set>
#ضمن <sstream>
#ضمن <string>

#جے_تعریف_شدہ _WIN32
#   define WIN32_LEAN_AND_اوسط
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#وإلا
#   include <dlfcn.h>
#   include <unistd.h>
#جے_انت

#ہدایت clang diagnostic ignored "-Wgnu-anonymous-struct"
#ہدایت clang diagnostic ignored "-Wmissing-prototypes"
#ہدایت clang diagnostic ignored "-Wsign-compare"
#ہدایت clang diagnostic ignored "-Wlanguage-extension-token"
#ہدایت clang diagnostic ignored "-Wmicrosoft-enum-value"
#ہدایت clang diagnostic ignored "-Wnested-anon-types"

#ضمن "ggml-impl.h"
#ضمن "htp-drv.h"
#ضمن "libdl.h"

#ضمن <domain.h>

//
// Driver API types
//

تعريفنوع فراغ * (*rpcmem_alloc_pfn_t)(صحيح heapid, uint32_t flags, صحيح size);
تعريفنوع فراغ * (*rpcmem_alloc2_pfn_t)(صحيح heapid, uint32_t flags, size_t size);
تعريفنوع فراغ   (*rpcmem_free_pfn_t)(فراغ * po);
تعريفنوع صحيح    (*rpcmem_منزل_fd_pfn_t)(فراغ * po);

تعريفنوع AEEResult (*dspqueue_create_pfn_t)(صحيح                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_قطار_ناپ,
                                           uint32_t            resp_قطار_ناپ,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           فراغ *              callback_context,
                                           dspqueue_t *        queue);
تعريفنوع AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
تعريفنوع AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *قطار_id);
تعريفنوع AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t عدد_buffers,
                                          بنية dspqueue_بفر *buffers,
                                          uint32_t سنیہا_لمائی,
                                          ثابت uint8_t *سنیہا,
                                          uint32_t timeout_us);
تعريفنوع AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *عدد_buffers,
                                         بنية dspqueue_بفر *buffers,
                                         uint32_t max_سنیہا_لمائی,
                                         uint32_t *سنیہا_لمائی, uint8_t *سنیہا,
                                         uint32_t timeout_us);
تعريفنوع AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *عدد_buffers,
                                                 بنية dspqueue_بفر *buffers,
                                                 uint32_t max_سنیہا_لمائی,
                                                 uint32_t *سنیہا_لمائی, uint8_t *سنیہا);
تعريفنوع صحيح (*fastrpc_mmap_pfn_t)(صحيح domain, صحيح fd, فراغ *addr, صحيح وتھ, size_t لمائی, تعداد fastrpc_map_flags flags);
تعريفنوع صحيح (*fastrpc_munmap_pfn_t)(صحيح domain, صحيح fd, فراغ *addr, size_t لمائی);

تعريفنوع صحيح (*remote_handle64_open_pfn_t)(ثابت حرف* name, remote_handle64 *ph);
تعريفنوع صحيح (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
تعريفنوع صحيح (*remote_handle64_close_pfn_t)(remote_handle h);
تعريفنوع صحيح (*remote_handle_control_pfn_t)(uint32_t req, فراغ* انکڑے, uint32_t datalen);
تعريفنوع صحيح (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, فراغ* انکڑے, uint32_t datalen);
تعريفنوع صحيح (*remote_session_control_pfn_t)(uint32_t req, فراغ *انکڑے, uint32_t datalen);
تعريفنوع صحيح (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = خالی_سوچک;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = خالی_سوچک;
rpcmem_free_pfn_t   rpcmem_free_pfn   = خالی_سوچک;
rpcmem_منزل_fd_pfn_t  rpcmem_منزل_fd_pfn  = خالی_سوچک;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = خالی_سوچک;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = خالی_سوچک;

dspqueue_create_pfn_t       dspqueue_create_pfn       = خالی_سوچک;
dspqueue_close_pfn_t        dspqueue_close_pfn        = خالی_سوچک;
dspqueue_export_pfn_t       dspqueue_export_pfn       = خالی_سوچک;
dspqueue_write_pfn_t        dspqueue_write_pfn        = خالی_سوچک;
dspqueue_read_pfn_t         dspqueue_read_pfn         = خالی_سوچک;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = خالی_سوچک;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = خالی_سوچک;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = خالی_سوچک;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = خالی_سوچک;
remote_handle_control_pfn_t   remote_handle_control_pfn   = خالی_سوچک;
remote_handle64_control_pfn_t remote_handle64_control_pfn = خالی_سوچک;
remote_session_control_pfn_t  remote_session_control_pfn  = خالی_سوچک;
remote_system_request_pfn_t   remote_system_request_pfn   = خالی_سوچک;

//
// Driver API
//

فراغ * rpcmem_alloc(صحيح heapid, uint32_t flags, صحيح size) {
    أرجع rpcmem_alloc_pfn(heapid, flags, size);
}

فراغ * rpcmem_alloc2(صحيح heapid, uint32_t flags, size_t size) {
    إذا (rpcmem_alloc2_pfn) {
        أرجع rpcmem_alloc2_pfn(heapid, flags, size);
    } وإلا {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        أرجع rpcmem_alloc_pfn(heapid, flags, size);
    }
}

فراغ rpcmem_free(فراغ * po) {
    أرجع rpcmem_free_pfn(po);
}

صحيح rpcmem_منزل_fd(فراغ * po) {
    أرجع rpcmem_منزل_fd_pfn(po);
}

HTPDRV_API صحيح fastrpc_mmap(صحيح domain, صحيح fd, فراغ * addr, صحيح وتھ, size_t لمائی, تعداد fastrpc_map_flags flags) {
    أرجع fastrpc_mmap_pfn(domain, fd, addr, وتھ, لمائی, flags);
}

HTPDRV_API صحيح fastrpc_munmap(صحيح domain, صحيح fd, فراغ * addr, size_t لمائی) {
    أرجع fastrpc_munmap_pfn(domain, fd, addr, لمائی);
}

AEEResult dspqueue_create(صحيح                 domain,
                          uint32_t            flags,
                          uint32_t            req_قطار_ناپ,
                          uint32_t            resp_قطار_ناپ,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          فراغ *              callback_context,
                          dspqueue_t *        queue) {
    أرجع dspqueue_create_pfn(domain, flags, req_قطار_ناپ, resp_قطار_ناپ, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    أرجع dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * قطار_id) {
    أرجع dspqueue_export_pfn(queue, قطار_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 عدد_buffers,
                         بنية dspqueue_بفر * buffers,
                         uint32_t                 سنیہا_لمائی,
                         ثابت uint8_t *          سنیہا,
                         uint32_t                 timeout_us) {
    أرجع dspqueue_write_pfn(queue, flags, عدد_buffers, buffers, سنیہا_لمائی, سنیہا, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               عدد_buffers,
                        بنية dspqueue_بفر * buffers,
                        uint32_t                 max_سنیہا_لمائی,
                        uint32_t *               سنیہا_لمائی,
                        uint8_t *                سنیہا,
                        uint32_t                 timeout_us) {
#جے_تعریف_شدہ _WIN32
    إذا (timeout_us == 0) {
        أرجع dspqueue_read_noblock_pfn(queue, flags, max_buffers, عدد_buffers, buffers, max_سنیہا_لمائی,
                                     سنیہا_لمائی, سنیہا);
    }
#جے_انت
    أرجع dspqueue_read_pfn(queue, flags, max_buffers, عدد_buffers, buffers, max_سنیہا_لمائی, سنیہا_لمائی,
                             سنیہا, timeout_us);
}

HTPDRV_API صحيح remote_handle64_open(ثابت حرف * name, remote_handle64 * ph) {
    أرجع remote_handle64_open_pfn(name, ph);
}

HTPDRV_API صحيح remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    أرجع remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API صحيح remote_handle64_close(remote_handle64 h) {
    أرجع remote_handle64_close_pfn(h);
}

HTPDRV_API صحيح remote_handle_control(uint32_t req, فراغ * انکڑے, uint32_t datalen) {
    أرجع remote_handle_control_pfn(req, انکڑے, datalen);
}

HTPDRV_API صحيح remote_handle64_control(remote_handle64 h, uint32_t req, فراغ * انکڑے, uint32_t datalen) {
    أرجع remote_handle64_control_pfn(h, req, انکڑے, datalen);
}

HTPDRV_API صحيح remote_session_control(uint32_t req, فراغ * انکڑے, uint32_t datalen) {
    أرجع remote_session_control_pfn(req, انکڑے, datalen);
}

HTPDRV_API صحيح remote_system_request(system_req_payload * req) {
    إذا (!remote_system_request_pfn) {
        أرجع AEE_EUNSUPPORTEDAPI;
    }
    أرجع remote_system_request_pfn(req);
}

#جے_تعریف_شدہ _WIN32

ساكن std::string wstr_منزل_str(std::wstring_view wstr) {
    std::string نتیجہ;
    إذا (wstr.خالی_تھاواں()) {
        أرجع نتیجہ;
    }
    خودکار بائٹاں_needed = Wide_Char_منزل_Multi_Byte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.انکڑے(), (صحيح) wstr.size(),
                                            خالی_سوچک, 0, خالی_سوچک, خالی_سوچک);
    إذا (بائٹاں_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", Get_آخری_Error());
        ارم std::runtime_error("Invalid wstring input");
    }

    نتیجہ.resize(بائٹاں_needed, '\0');
    صحيح بائٹاں_written = Wide_Char_منزل_Multi_Byte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.انکڑے(), (صحيح) wstr.size(),
                                            نتیجہ.انکڑے(), بائٹاں_needed,
                                            خالی_سوچک, خالی_سوچک);
    إذا (بائٹاں_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", Get_آخری_Error());
        ارم std::runtime_error("Wstring conversion failed");
    }
    أرجع نتیجہ;
}

ساكن std::string get_driver_رستہ() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string نتیجہ;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(عدم, عدم, STANDARD_RIGHTS_READ);
    إذا (خالی_سوچک == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", Get_آخری_Error());
        أرجع نتیجہ;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    إذا (خالی_سوچک == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", Get_آخری_Error());
        CloseServiceHandle(schSCManager);
        أرجع نتیجہ;
    }

    // Store the size of buffer used as an output.
    DWORD بفر_ناپ_2;
    إذا (!QueryServiceConfigW(schService, عدم, 0, &بفر_ناپ_2) &&
        (Get_آخری_Error() != ERROR_INSUFFICIENT_بفر)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", Get_آخری_Error());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        أرجع نتیجہ;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, بفر_ناپ_2));
    إذا (!QueryServiceConfigW(schService, serviceConfig, بفر_ناپ_2, &بفر_ناپ_2)) {
        fprintf(stderr, "ggml-hex: Failed to query service config. Error: %lu\n", Get_آخری_Error());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        أرجع نتیجہ;
    }

    // Read the driver file path get its parent directory
    std::wstring driver_رستہ = std::wstring(serviceConfig->lp_ثنائی_رستہ_Name);
    driver_رستہ = driver_رستہ.substr(0, driver_رستہ.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    ثابت std::wstring system_جڑ_Placeholder = L"\\SystemRoot";
    إذا (0 != driver_رستہ.compare(0, system_جڑ_Placeholder.لمائی(), system_جڑ_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        أرجع نتیجہ;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    ثابت std::wstring system_جڑ_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD عدد_الفاظ = GetEnvironmentVariableW(system_جڑ_Env.c_str(), عدم, 0);
    إذا (عدد_الفاظ == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        أرجع نتیجہ;
    }

    // Query the actual system root name from environment variable
    std::vector<wchar_t> system_جڑ(عدد_الفاظ + 1);
    عدد_الفاظ = GetEnvironmentVariableW(system_جڑ_Env.c_str(), system_جڑ.انکڑے(), عدد_الفاظ + 1);
    إذا (عدد_الفاظ == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        أرجع نتیجہ;
    }
    driver_رستہ.replace(0, system_جڑ_Placeholder.لمائی(), std::wstring(system_جڑ.انکڑے()));

    أرجع wstr_منزل_str(driver_رستہ);
}

#جے_انت

ورتوں dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

صحيح htpdrv_init() {
    ساكن dl_handle_ptr lib_cdsp_rpc_handle = خالی_سوچک;
    ساكن منطقي initialized = خطأ;
#جے_تعریف_شدہ _WIN32
    std::string drv_رستہ = get_driver_رستہ() + "\\" + "libcdsprpc.dll";
#وإلا
    std::string drv_رستہ = "libcdsprpc.so";
#جے_انت
    إذا (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        أرجع AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_رستہ.c_str());

    fs::path path{ drv_رستہ.c_str() };
    dl_handle_ptr handle { dl_لدو_library(path) };
    إذا (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        أرجع AEE_EUNABLETOLOAD;
    }

#تعریف_کرو dlsym(drv, type, pfn, علامت, ignore)                               \
    افعل {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        إذا (!ignore && خالی_سوچک == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            أرجع AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } طالما (0)

    dlsym(handle.get(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, خطأ);
    dlsym(handle.get(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, سچ);
    dlsym(handle.get(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, خطأ);
    dlsym(handle.get(), rpcmem_منزل_fd_pfn_t, rpcmem_منزل_fd_pfn, rpcmem_منزل_fd, خطأ);
    dlsym(handle.get(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, خطأ);
    dlsym(handle.get(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, خطأ);
    dlsym(handle.get(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, خطأ);
    dlsym(handle.get(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, خطأ);
    dlsym(handle.get(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, خطأ);
    dlsym(handle.get(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, خطأ);
    dlsym(handle.get(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, خطأ);
    dlsym(handle.get(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, خطأ);
    dlsym(handle.get(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, خطأ);
    dlsym(handle.get(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, خطأ);
    dlsym(handle.get(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, خطأ);
    dlsym(handle.get(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, خطأ);
    dlsym(handle.get(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, خطأ);
    dlsym(handle.get(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, خطأ);
    dlsym(handle.get(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, سچ);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = سچ;

    أرجع AEE_SUCCESS;
}

domain * htpdrv_get_domain(صحيح domain_id) {
    صحيح i    = 0;
    صحيح size = حجم(supported_domains) / حجم(domain);

    لكل (i = 0; i < size; i++) {
        إذا (supported_domains[i].id == domain_id) {
            أرجع &supported_domains[i];
        }
    }

    أرجع عدم;
}

صحيح htpdrv_get_arch(صحيح domain, صحيح * arch) {
    إذا (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        أرجع AEE_EUNSUPPORTEDAPI;
    }

    بنية remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    صحيح err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, حجم(arch_ver));
    إذا ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        أرجع AEE_EUNSUPPORTEDAPI;
    }

    إذا (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        أرجع err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (صحيح) ((val >> 4) * 10 + (val & 0x0f));
    أرجع 0;
}
