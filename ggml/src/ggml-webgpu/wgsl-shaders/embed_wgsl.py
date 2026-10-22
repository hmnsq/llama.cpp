استورد os
استورد re
استورد argparse


وصف expand_includes(shader, داخلا_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_نمونو = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    وصف replacer(ميلاپ):
        fname = ميلاپ.گروهه(1)
        file_رستو = os.رستو.join(داخلا_dir, fname)
        إذا ليس os.رستو.exists(file_رستو):
            ارفع File_Not_مليو_Error(f"Included file not found: {file_path}")
        سان open(file_رستو, "r", encoding="utf-8") كـ f:
            included_ڪوڊ = f.اقرأ()
        # Recursively expand includes inside the included file
        أرجع expand_includes(included_ڪوڊ, داخلا_dir)

    أرجع include_نمونو.sub(replacer, shader)


وصف chunk_shader(shader_ڪوڊ, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    أرجع [shader_ڪوڊ[i : i + max_chunk_len] لكل i في سلسلو(0, ڊيگهه(shader_ڪوڊ), max_chunk_len)]


وصف raw_delim(shader_ڪوڊ):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    طالما f"){delim}\"" في shader_ڪوڊ:
        delim += "_x"
    أرجع delim


وصف write_shader(shader_name, shader_ڪوڊ, output_dir, outfile, داخلا_dir):
    shader_ڪوڊ = expand_includes(shader_ڪوڊ, داخلا_dir)

    إذا output_dir:
        wgsl_filename = os.رستو.join(output_dir, f"{shader_name}.wgsl")
        سان open(wgsl_filename, "w", encoding="utf-8") كـ f_ٻاهريون:
            f_ٻاهريون.اكتب(shader_ڪوڊ)

    delim = raw_delim(shader_ڪوڊ)
    chunks = chunk_shader(shader_ڪوڊ)

    إذا ڊيگهه(chunks) == 1:
        outfile.اكتب(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    وإلا:
        لكل idx, chunk في enumerate(chunks):
            outfile.اكتب(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.اكتب(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.اكتب('    static const std::string s = []{\n')
        outfile.اكتب('        std::string tmp;\n')
        outfile.اكتب(f'        tmp.reserve({len(shader_code)});\n')
        لكل idx في سلسلو(ڊيگهه(chunks)):
            outfile.اكتب(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.اكتب('        return tmp;\n')
        outfile.اكتب('    }();\n')
        outfile.اكتب('    return s;\n')
        outfile.اكتب('}\n')
        outfile.اكتب(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


وصف مکيه():
    parser = argparse.دليل_Parser()
    parser.جوڙيو_دليل("--input_dir", required=سچ)
    parser.جوڙيو_دليل("--output_file", required=سچ)
    دليلون_2 = parser.parse_دليلون()

    سان open(دليلون_2.output_file, "w", encoding="utf-8") كـ ٻاهريون:
        ٻاهريون.اكتب("// Auto-generated shader embedding\n")
        ٻاهريون.اكتب("#include <string>\n\n")
        لكل fname في sorted(os.listdir(دليلون_2.داخلا_dir)):
            إذا fname.endswith(".wgsl"):
                shader_رستو = os.رستو.join(دليلون_2.داخلا_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                سان open(shader_رستو, "r", encoding="utf-8") كـ f:
                    shader_ڪوڊ = f.اقرأ()

                write_shader(shader_name, shader_ڪوڊ, ڪجهه_نه, ٻاهريون, دليلون_2.داخلا_dir)


إذا __name__ == "__main__":
    مکيه()
