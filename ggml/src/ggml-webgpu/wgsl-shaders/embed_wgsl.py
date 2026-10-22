استورد os
استورد re
استورد argparse


تعریف expand_includes(shader, داخلہ_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_نمونہ = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    تعریف replacer(میل):
        fname = میل.گروہ(1)
        file_رستہ = os.رستہ.join(داخلہ_dir, fname)
        إذا ليس os.رستہ.exists(file_رستہ):
            ارفع File_Not_لبھیا_Error(f"Included file not found: {file_path}")
        نال open(file_رستہ, "r", encoding="utf-8") كـ f:
            included_کوڈ = f.اقرأ()
        # Recursively expand includes inside the included file
        أرجع expand_includes(included_کوڈ, داخلہ_dir)

    أرجع include_نمونہ.sub(replacer, shader)


تعریف chunk_shader(shader_کوڈ, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    أرجع [shader_کوڈ[i : i + max_chunk_len] لكل i في سلسلہ(0, لمبائی(shader_کوڈ), max_chunk_len)]


تعریف raw_delim(shader_کوڈ):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    طالما f"){delim}\"" في shader_کوڈ:
        delim += "_x"
    أرجع delim


تعریف write_shader(shader_name, shader_کوڈ, output_dir, outfile, داخلہ_dir):
    shader_کوڈ = expand_includes(shader_کوڈ, داخلہ_dir)

    إذا output_dir:
        wgsl_filename = os.رستہ.join(output_dir, f"{shader_name}.wgsl")
        نال open(wgsl_filename, "w", encoding="utf-8") كـ f_باہرلا:
            f_باہرلا.اكتب(shader_کوڈ)

    delim = raw_delim(shader_کوڈ)
    chunks = chunk_shader(shader_کوڈ)

    إذا لمبائی(chunks) == 1:
        outfile.اكتب(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    وإلا:
        لكل idx, chunk في enumerate(chunks):
            outfile.اكتب(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.اكتب(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.اكتب('    static const std::string s = []{\n')
        outfile.اكتب('        std::string tmp;\n')
        outfile.اكتب(f'        tmp.reserve({len(shader_code)});\n')
        لكل idx في سلسلہ(لمبائی(chunks)):
            outfile.اكتب(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.اكتب('        return tmp;\n')
        outfile.اكتب('    }();\n')
        outfile.اكتب('    return s;\n')
        outfile.اكتب('}\n')
        outfile.اكتب(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


تعریف مکھ():
    parser = argparse.دلیل_Parser()
    parser.جوڑو_دلیل("--input_dir", required=سچ)
    parser.جوڑو_دلیل("--output_file", required=سچ)
    دلیلاں_2 = parser.parse_دلیلاں()

    نال open(دلیلاں_2.output_file, "w", encoding="utf-8") كـ باہرلا:
        باہرلا.اكتب("// Auto-generated shader embedding\n")
        باہرلا.اكتب("#include <string>\n\n")
        لكل fname في sorted(os.listdir(دلیلاں_2.داخلہ_dir)):
            إذا fname.endswith(".wgsl"):
                shader_رستہ = os.رستہ.join(دلیلاں_2.داخلہ_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                نال open(shader_رستہ, "r", encoding="utf-8") كـ f:
                    shader_کوڈ = f.اقرأ()

                write_shader(shader_name, shader_کوڈ, کجھ_نئیں, باہرلا, دلیلاں_2.داخلہ_dir)


إذا __name__ == "__main__":
    مکھ()
