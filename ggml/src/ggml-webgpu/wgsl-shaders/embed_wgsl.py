استورد os
استورد re
استورد argparse


تعریف expand_includes(shader, درون_داد_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_الگو = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    تعریف replacer(تطابق):
        fname = تطابق.گروه(1)
        file_مسیر = os.مسیر.join(درون_داد_dir, fname)
        إذا ليس os.مسیر.exists(file_مسیر):
            ارفع File_Not_یافت_Error(f"Included file not found: {file_path}")
        با open(file_مسیر, "r", encoding="utf-8") كـ f:
            included_کد = f.اقرأ()
        # Recursively expand includes inside the included file
        أرجع expand_includes(included_کد, درون_داد_dir)

    أرجع include_الگو.sub(replacer, shader)


تعریف chunk_shader(shader_کد, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    أرجع [shader_کد[i : i + max_chunk_len] لكل i في دامنه(0, طول(shader_کد), max_chunk_len)]


تعریف raw_delim(shader_کد):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    طالما f"){delim}\"" في shader_کد:
        delim += "_x"
    أرجع delim


تعریف write_shader(shader_name, shader_کد, output_dir, outfile, درون_داد_dir):
    shader_کد = expand_includes(shader_کد, درون_داد_dir)

    إذا output_dir:
        wgsl_filename = os.مسیر.join(output_dir, f"{shader_name}.wgsl")
        با open(wgsl_filename, "w", encoding="utf-8") كـ f_بیرون:
            f_بیرون.اكتب(shader_کد)

    delim = raw_delim(shader_کد)
    chunks = chunk_shader(shader_کد)

    إذا طول(chunks) == 1:
        outfile.اكتب(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    وإلا:
        لكل idx, chunk في enumerate(chunks):
            outfile.اكتب(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.اكتب(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.اكتب('    static const std::string s = []{\n')
        outfile.اكتب('        std::string tmp;\n')
        outfile.اكتب(f'        tmp.reserve({len(shader_code)});\n')
        لكل idx في دامنه(طول(chunks)):
            outfile.اكتب(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.اكتب('        return tmp;\n')
        outfile.اكتب('    }();\n')
        outfile.اكتب('    return s;\n')
        outfile.اكتب('}\n')
        outfile.اكتب(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


تعریف اصلی():
    parser = argparse.آرگومان_Parser()
    parser.جمع_کن_آرگومان("--input_dir", required=درست)
    parser.جمع_کن_آرگومان("--output_file", required=درست)
    آرگومانها_2 = parser.parse_آرگومانها()

    با open(آرگومانها_2.output_file, "w", encoding="utf-8") كـ بیرون:
        بیرون.اكتب("// Auto-generated shader embedding\n")
        بیرون.اكتب("#include <string>\n\n")
        لكل fname في sorted(os.listdir(آرگومانها_2.درون_داد_dir)):
            إذا fname.endswith(".wgsl"):
                shader_مسیر = os.مسیر.join(آرگومانها_2.درون_داد_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                با open(shader_مسیر, "r", encoding="utf-8") كـ f:
                    shader_کد = f.اقرأ()

                write_shader(shader_name, shader_کد, هیچ, بیرون, آرگومانها_2.درون_داد_dir)


إذا __name__ == "__main__":
    اصلی()
