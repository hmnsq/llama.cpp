#समावेश <filesystem>
#समावेश <set>
#समावेश <sstream>
#समावेश <string>

#अगर_घोषित _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#अन्यथा
#   include <dlfcn.h>
#   include <unistd.h>
#पूर्ण_अगर

#आशय clang diagnostic ignored "-Wgnu-anonymous-struct"
#आशय clang diagnostic ignored "-Wmissing-prototypes"
#आशय clang diagnostic ignored "-Wsign-compare"
#आशय clang diagnostic ignored "-Wlanguage-extension-token"
#आशय clang diagnostic ignored "-Wmicrosoft-enum-value"
#आशय clang diagnostic ignored "-Wnested-anon-types"

#समावेश "ggml-impl.h"
#समावेश "htp-drv.h"
#समावेश "libdl.h"

#समावेश <domain.h>

//
// Driver API types
//

प्रकार व्योम * (*rpcmem_alloc_pfn_t)(पूर्णांक heapid, uint32_t flags, पूर्णांक size);
प्रकार व्योम * (*rpcmem_alloc2_pfn_t)(पूर्णांक heapid, uint32_t flags, माप_प्रकार size);
प्रकार व्योम   (*rpcmem_free_pfn_t)(व्योम * po);
प्रकार पूर्णांक    (*rpcmem_to_fd_pfn_t)(व्योम * po);

प्रकार AEEResult (*dspqueue_create_pfn_t)(पूर्णांक                 domain,
                                           uint32_t            flags,
                                           uint32_t            req_queue_आकार,
                                           uint32_t            resp_queue_आकार,
                                           dspqueue_callback_t packet_callback,
                                           dspqueue_callback_t error_callback,
                                           व्योम *              callback_context,
                                           dspqueue_t *        queue);
प्रकार AEEResult (*dspqueue_close_pfn_t)(dspqueue_t queue);
प्रकार AEEResult (*dspqueue_export_pfn_t)(dspqueue_t queue, uint64_t *queue_id);
प्रकार AEEResult (*dspqueue_write_pfn_t)(dspqueue_t queue, uint32_t flags,
                                          uint32_t अंक_buffers,
                                          काष्ठा dspqueue_बफर *buffers,
                                          uint32_t message_लंबाई,
                                          स्थिर uint8_t *message,
                                          uint32_t timeout_us);
प्रकार AEEResult (*dspqueue_read_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                         uint32_t max_buffers, uint32_t *अंक_buffers,
                                         काष्ठा dspqueue_बफर *buffers,
                                         uint32_t max_message_लंबाई,
                                         uint32_t *message_लंबाई, uint8_t *message,
                                         uint32_t timeout_us);
प्रकार AEEResult (*dspqueue_read_noblock_pfn_t)(dspqueue_t queue, uint32_t *flags,
                                                 uint32_t max_buffers, uint32_t *अंक_buffers,
                                                 काष्ठा dspqueue_बफर *buffers,
                                                 uint32_t max_message_लंबाई,
                                                 uint32_t *message_लंबाई, uint8_t *message);
प्रकार पूर्णांक (*fastrpc_mmap_pfn_t)(पूर्णांक domain, पूर्णांक fd, व्योम *addr, पूर्णांक offset, माप_प्रकार लंबाई, क्रमागत fastrpc_map_flags flags);
प्रकार पूर्णांक (*fastrpc_munmap_pfn_t)(पूर्णांक domain, पूर्णांक fd, व्योम *addr, माप_प्रकार लंबाई);

प्रकार पूर्णांक (*remote_handle64_open_pfn_t)(स्थिर अक्षर* name, remote_handle64 *ph);
प्रकार पूर्णांक (*remote_handle64_invoke_pfn_t)(remote_handle64 h, uint32_t dwScalars, remote_arg *pra);
प्रकार पूर्णांक (*remote_handle64_close_pfn_t)(remote_handle h);
प्रकार पूर्णांक (*remote_handle_control_pfn_t)(uint32_t req, व्योम* तथ्य, uint32_t datalen);
प्रकार पूर्णांक (*remote_handle64_control_pfn_t)(remote_handle64 h, uint32_t req, व्योम* तथ्य, uint32_t datalen);
प्रकार पूर्णांक (*remote_session_control_pfn_t)(uint32_t req, व्योम *तथ्य, uint32_t datalen);
प्रकार पूर्णांक (*remote_system_request_pfn_t)(system_req_payload * req);

//
// Driver API pfns
//

rpcmem_alloc_pfn_t  rpcmem_alloc_pfn  = nullptr;
rpcmem_alloc2_pfn_t rpcmem_alloc2_pfn = nullptr;
rpcmem_free_pfn_t   rpcmem_free_pfn   = nullptr;
rpcmem_to_fd_pfn_t  rpcmem_to_fd_pfn  = nullptr;

fastrpc_mmap_pfn_t   fastrpc_mmap_pfn   = nullptr;
fastrpc_munmap_pfn_t fastrpc_munmap_pfn = nullptr;

dspqueue_create_pfn_t       dspqueue_create_pfn       = nullptr;
dspqueue_close_pfn_t        dspqueue_close_pfn        = nullptr;
dspqueue_export_pfn_t       dspqueue_export_pfn       = nullptr;
dspqueue_write_pfn_t        dspqueue_write_pfn        = nullptr;
dspqueue_read_pfn_t         dspqueue_read_pfn         = nullptr;
dspqueue_read_noblock_pfn_t dspqueue_read_noblock_pfn = nullptr;

remote_handle64_open_pfn_t    remote_handle64_open_pfn    = nullptr;
remote_handle64_invoke_pfn_t  remote_handle64_invoke_pfn  = nullptr;
remote_handle64_close_pfn_t   remote_handle64_close_pfn   = nullptr;
remote_handle_control_pfn_t   remote_handle_control_pfn   = nullptr;
remote_handle64_control_pfn_t remote_handle64_control_pfn = nullptr;
remote_session_control_pfn_t  remote_session_control_pfn  = nullptr;
remote_system_request_pfn_t   remote_system_request_pfn   = nullptr;

//
// Driver API
//

व्योम * rpcmem_alloc(पूर्णांक heapid, uint32_t flags, पूर्णांक size) {
    वापस rpcmem_alloc_pfn(heapid, flags, size);
}

व्योम * rpcmem_alloc2(पूर्णांक heapid, uint32_t flags, माप_प्रकार size) {
    अगर (rpcmem_alloc2_pfn) {
        वापस rpcmem_alloc2_pfn(heapid, flags, size);
    } अन्यथा {
        GGML_LOG_INFO("ggml-hex: rpcmem_alloc2 not found, falling back to rpcmem_alloc\n");
        वापस rpcmem_alloc_pfn(heapid, flags, size);
    }
}

व्योम rpcmem_free(व्योम * po) {
    वापस rpcmem_free_pfn(po);
}

पूर्णांक rpcmem_to_fd(व्योम * po) {
    वापस rpcmem_to_fd_pfn(po);
}

HTPDRV_API पूर्णांक fastrpc_mmap(पूर्णांक domain, पूर्णांक fd, व्योम * addr, पूर्णांक offset, माप_प्रकार लंबाई, क्रमागत fastrpc_map_flags flags) {
    वापस fastrpc_mmap_pfn(domain, fd, addr, offset, लंबाई, flags);
}

HTPDRV_API पूर्णांक fastrpc_munmap(पूर्णांक domain, पूर्णांक fd, व्योम * addr, माप_प्रकार लंबाई) {
    वापस fastrpc_munmap_pfn(domain, fd, addr, लंबाई);
}

AEEResult dspqueue_create(पूर्णांक                 domain,
                          uint32_t            flags,
                          uint32_t            req_queue_आकार,
                          uint32_t            resp_queue_आकार,
                          dspqueue_callback_t packet_callback,
                          dspqueue_callback_t error_callback,
                          व्योम *              callback_context,
                          dspqueue_t *        queue) {
    वापस dspqueue_create_pfn(domain, flags, req_queue_आकार, resp_queue_आकार, packet_callback, error_callback,
                               callback_context, queue);
}

AEEResult dspqueue_close(dspqueue_t queue) {
    वापस dspqueue_close_pfn(queue);
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    वापस dspqueue_export_pfn(queue, queue_id);
}

AEEResult dspqueue_write(dspqueue_t               queue,
                         uint32_t                 flags,
                         uint32_t                 अंक_buffers,
                         काष्ठा dspqueue_बफर * buffers,
                         uint32_t                 message_लंबाई,
                         स्थिर uint8_t *          message,
                         uint32_t                 timeout_us) {
    वापस dspqueue_write_pfn(queue, flags, अंक_buffers, buffers, message_लंबाई, message, timeout_us);
}

AEEResult dspqueue_read(dspqueue_t               queue,
                        uint32_t *               flags,
                        uint32_t                 max_buffers,
                        uint32_t *               अंक_buffers,
                        काष्ठा dspqueue_बफर * buffers,
                        uint32_t                 max_message_लंबाई,
                        uint32_t *               message_लंबाई,
                        uint8_t *                message,
                        uint32_t                 timeout_us) {
#अगर_घोषित _WIN32
    अगर (timeout_us == 0) {
        वापस dspqueue_read_noblock_pfn(queue, flags, max_buffers, अंक_buffers, buffers, max_message_लंबाई,
                                     message_लंबाई, message);
    }
#पूर्ण_अगर
    वापस dspqueue_read_pfn(queue, flags, max_buffers, अंक_buffers, buffers, max_message_लंबाई, message_लंबाई,
                             message, timeout_us);
}

HTPDRV_API पूर्णांक remote_handle64_open(स्थिर अक्षर * name, remote_handle64 * ph) {
    वापस remote_handle64_open_pfn(name, ph);
}

HTPDRV_API पूर्णांक remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra) {
    वापस remote_handle64_invoke_pfn(h, dwScalars, pra);
}

HTPDRV_API पूर्णांक remote_handle64_close(remote_handle64 h) {
    वापस remote_handle64_close_pfn(h);
}

HTPDRV_API पूर्णांक remote_handle_control(uint32_t req, व्योम * तथ्य, uint32_t datalen) {
    वापस remote_handle_control_pfn(req, तथ्य, datalen);
}

HTPDRV_API पूर्णांक remote_handle64_control(remote_handle64 h, uint32_t req, व्योम * तथ्य, uint32_t datalen) {
    वापस remote_handle64_control_pfn(h, req, तथ्य, datalen);
}

HTPDRV_API पूर्णांक remote_session_control(uint32_t req, व्योम * तथ्य, uint32_t datalen) {
    वापस remote_session_control_pfn(req, तथ्य, datalen);
}

HTPDRV_API पूर्णांक remote_system_request(system_req_payload * req) {
    अगर (!remote_system_request_pfn) {
        वापस AEE_EUNSUPPORTEDAPI;
    }
    वापस remote_system_request_pfn(req);
}

#अगर_घोषित _WIN32

अटल std::string wstr_to_str(std::wstring_view wstr) {
    std::string परिणाम;
    अगर (wstr.empty()) {
        वापस परिणाम;
    }
    स्वतः bytes_needed = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.तथ्य(), (पूर्णांक) wstr.size(),
                                            nullptr, 0, nullptr, nullptr);
    अगर (bytes_needed == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        फेंको std::runtime_error("Invalid wstring input");
    }

    परिणाम.resize(bytes_needed, '\0');
    पूर्णांक bytes_written = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                            wstr.तथ्य(), (पूर्णांक) wstr.size(),
                                            परिणाम.तथ्य(), bytes_needed,
                                            nullptr, nullptr);
    अगर (bytes_written == 0) {
        GGML_LOG_ERROR("ggml-hex: WideCharToMultiByte failed. Error %lu\n", GetLastError());
        फेंको std::runtime_error("Wstring conversion failed");
    }
    वापस परिणाम;
}

अटल std::string get_driver_path() {
    std::wstring serviceName = L"qcnspmcdm";
    std::string परिणाम;

    // Get a handle to the SCM database.
    SC_HANDLE schSCManager = OpenSCManagerW(शून्य, शून्य, STANDARD_RIGHTS_READ);
    अगर (nullptr == schSCManager) {
        GGML_LOG_ERROR("ggml-hex: Failed to open SCManager. Error: %lu\n", GetLastError());
        वापस परिणाम;
    }

    // Get a handle to the service.
    SC_HANDLE schService = OpenServiceW(schSCManager,           // SCM database
                                        serviceName.c_str(),    // name of service
                                        SERVICE_QUERY_CONFIG);  // need query config access

    अगर (nullptr == schService) {
        GGML_LOG_ERROR("ggml-hex: Failed to open qcnspmcdm service. Error: %lu\n", GetLastError());
        CloseServiceHandle(schSCManager);
        वापस परिणाम;
    }

    // Store the size of buffer used as an output.
    DWORD बफर_आकार_2;
    अगर (!QueryServiceConfigW(schService, शून्य, 0, &बफर_आकार_2) &&
        (GetLastError() != ERROR_INSUFFICIENT_बफर)) {
        GGML_LOG_ERROR("ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        वापस परिणाम;
    }
    // Get the configuration of the service.
    LPQUERY_SERVICE_CONFIGW serviceConfig =
        static_cast<LPQUERY_SERVICE_CONFIGW>(LocalAlloc(LMEM_FIXED, बफर_आकार_2));
    अगर (!QueryServiceConfigW(schService, serviceConfig, बफर_आकार_2, &बफर_आकार_2)) {
        ख_लिखो(मानक_त्रुटि, "ggml-hex: Failed to query service config. Error: %lu\n", GetLastError());
        LocalFree(serviceConfig);
        CloseServiceHandle(schService);
        CloseServiceHandle(schSCManager);
        वापस परिणाम;
    }

    // Read the driver file path get its parent directory
    std::wstring driverPath = std::wstring(serviceConfig->lp_द्विआधारी_Path_Name);
    driverPath = driverPath.substr(0, driverPath.find_last_of(L"\\"));

    // Clean up resources
    LocalFree(serviceConfig);
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);

    // Driver path would contain invalid path string, like:
    // \SystemRoot\System32\DriverStore\FileRepository\qcadsprpc8280.inf_arm64_c2b9460c9a072f37
    // "\SystemRoot" should be replace with a correct one (e.g. C:\Windows)
    स्थिर std::wstring system_मूल_Placeholder = L"\\SystemRoot";
    अगर (0 != driverPath.compare(0, system_मूल_Placeholder.लंबाई(), system_मूल_Placeholder)) {
        GGML_LOG_ERROR("ggml-hex: String pattern not found in driver path.\n");
        वापस परिणाम;
    }

    // Replace \SystemRoot with an absolute path from system ENV windir
    स्थिर std::wstring system_मूल_Env = L"windir";

    // Query the number of wide characters this variable requires
    DWORD अंक_शब्दगण = GetEnvironmentVariableW(system_मूल_Env.c_str(), शून्य, 0);
    अगर (अंक_शब्दगण == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed get systemRoot environment variable\n");
        वापस परिणाम;
    }

    // Query the actual system root name from environment variable
    std::vector<ब_अक्षर_प्रकार> system_मूल(अंक_शब्दगण + 1);
    अंक_शब्दगण = GetEnvironmentVariableW(system_मूल_Env.c_str(), system_मूल.तथ्य(), अंक_शब्दगण + 1);
    अगर (अंक_शब्दगण == 0) {
        GGML_LOG_ERROR("ggml-hex: Failed to read windir environment variable\n");
        वापस परिणाम;
    }
    driverPath.replace(0, system_मूल_Placeholder.लंबाई(), std::wstring(system_मूल.तथ्य()));

    वापस wstr_to_str(driverPath);
}

#पूर्ण_अगर

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

पूर्णांक htpdrv_init() {
    अटल dl_handle_ptr lib_cdsp_rpc_handle = nullptr;
    अटल bool initialized = false;
#अगर_घोषित _WIN32
    std::string drv_path = get_driver_path() + "\\" + "libcdsprpc.dll";
#अन्यथा
    std::string drv_path = "libcdsprpc.so";
#पूर्ण_अगर
    अगर (initialized) {
        GGML_LOG_INFO("ggml-hex: Driver already loaded\n");
        वापस AEE_SUCCESS;
    }
    GGML_LOG_INFO("ggml-hex: Loading driver %s\n", drv_path.c_str());

    fs::path path{ drv_path.c_str() };
    dl_handle_ptr handle { dl_load_library(path) };
    अगर (!handle) {
        GGML_LOG_ERROR("ggml-hex: failed to load %s: %s\n", path.u8string().c_str(), dl_error());
        वापस AEE_EUNABLETOLOAD;
    }

#घोषणा dlsym(drv, type, pfn, प्रतीक, ignore)                               \
    करो {                                                                    \
        pfn = (type) dl_get_sym(drv, #symbol);                              \
        अगर (!ignore && nullptr == pfn) {                                    \
            GGML_LOG_ERROR("ggml-hex: failed to dlsym %s\n", #symbol);      \
            वापस AEE_EUNABLETOLOAD;                                       \
        }                                                                   \
    } जबतक (0)

    dlsym(handle.लो(), rpcmem_alloc_pfn_t, rpcmem_alloc_pfn, rpcmem_alloc, false);
    dlsym(handle.लो(), rpcmem_alloc2_pfn_t, rpcmem_alloc2_pfn, rpcmem_alloc2, true);
    dlsym(handle.लो(), rpcmem_free_pfn_t, rpcmem_free_pfn, rpcmem_free, false);
    dlsym(handle.लो(), rpcmem_to_fd_pfn_t, rpcmem_to_fd_pfn, rpcmem_to_fd, false);
    dlsym(handle.लो(), fastrpc_mmap_pfn_t, fastrpc_mmap_pfn, fastrpc_mmap, false);
    dlsym(handle.लो(), fastrpc_munmap_pfn_t, fastrpc_munmap_pfn, fastrpc_munmap, false);
    dlsym(handle.लो(), dspqueue_create_pfn_t, dspqueue_create_pfn, dspqueue_create, false);
    dlsym(handle.लो(), dspqueue_close_pfn_t, dspqueue_close_pfn, dspqueue_close, false);
    dlsym(handle.लो(), dspqueue_export_pfn_t, dspqueue_export_pfn, dspqueue_export, false);
    dlsym(handle.लो(), dspqueue_write_pfn_t, dspqueue_write_pfn, dspqueue_write, false);
    dlsym(handle.लो(), dspqueue_read_pfn_t, dspqueue_read_pfn, dspqueue_read, false);
    dlsym(handle.लो(), dspqueue_read_noblock_pfn_t, dspqueue_read_noblock_pfn, dspqueue_read_noblock, false);
    dlsym(handle.लो(), remote_handle64_open_pfn_t, remote_handle64_open_pfn, remote_handle64_open, false);
    dlsym(handle.लो(), remote_handle64_invoke_pfn_t, remote_handle64_invoke_pfn, remote_handle64_invoke, false);
    dlsym(handle.लो(), remote_handle_control_pfn_t, remote_handle_control_pfn, remote_handle_control, false);
    dlsym(handle.लो(), remote_handle64_control_pfn_t, remote_handle64_control_pfn, remote_handle64_control, false);
    dlsym(handle.लो(), remote_session_control_pfn_t, remote_session_control_pfn, remote_session_control, false);
    dlsym(handle.लो(), remote_handle64_close_pfn_t, remote_handle64_close_pfn, remote_handle64_close, false);
    dlsym(handle.लो(), remote_system_request_pfn_t, remote_system_request_pfn, remote_system_request, true);

    lib_cdsp_rpc_handle = std::move(handle);
    initialized         = true;

    वापस AEE_SUCCESS;
}

domain * htpdrv_get_domain(पूर्णांक domain_id) {
    पूर्णांक i    = 0;
    पूर्णांक size = माप(supported_domains) / माप(domain);

    क्रम (i = 0; i < size; i++) {
        अगर (supported_domains[i].id == domain_id) {
            वापस &supported_domains[i];
        }
    }

    वापस शून्य;
}

पूर्णांक htpdrv_get_arch(पूर्णांक domain, पूर्णांक * arch) {
    अगर (!remote_handle_control_pfn) {
        GGML_LOG_ERROR("ggml-hex: remote_handle_control is not supported on this device\n");
        वापस AEE_EUNSUPPORTEDAPI;
    }

    काष्ठा remote_dsp_capability arch_ver;
    arch_ver.domain       = (uint32_t) domain;
    arch_ver.attribute_ID = ARCH_VER;
    arch_ver.capability   = (uint32_t) 0;

    पूर्णांक err = remote_handle_control(DSPRPC_GET_DSP_INFO, &arch_ver, माप(arch_ver));
    अगर ((err & 0xff) == (AEE_EUNSUPPORTEDAPI & 0xff)) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability API is not supported on this device\n");
        वापस AEE_EUNSUPPORTEDAPI;
    }

    अगर (err != AEE_SUCCESS) {
        GGML_LOG_ERROR("ggml-hex: FastRPC capability query failed (err %d)\n", err);
        वापस err;
    }

    uint32_t val = arch_ver.capability & 0xff;
    *arch = (पूर्णांक) ((val >> 4) * 10 + (val & 0x0f));
    वापस 0;
}
