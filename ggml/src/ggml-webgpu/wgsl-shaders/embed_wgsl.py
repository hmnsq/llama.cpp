আমদানি os
আমদানি re
আমদানি argparse


def expand_includes(shader, প্রবেশ_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(প্রবেশ_dir, fname)
        যদি নয় os.path.exists(file_path):
            তোলো FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") ইব f:
            included_code = f.পঠ()
        # Recursively expand includes inside the included file
        ফেরত expand_includes(included_code, প্রবেশ_dir)

    ফেরত include_pattern.sub(replacer, shader)


def chunk_shader(shader_code, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    ফেরত [shader_code[i : i + max_chunk_len] জন্য i মধ্যে range(0, len(shader_code), max_chunk_len)]


def raw_delim(shader_code):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    যখন f"){delim}\"" মধ্যে shader_code:
        delim += "_x"
    ফেরত delim


def write_shader(shader_name, shader_code, output_dir, outfile, প্রবেশ_dir):
    shader_code = expand_includes(shader_code, প্রবেশ_dir)

    যদি output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") ইব f_out:
            f_out.লিখ(shader_code)

    delim = raw_delim(shader_code)
    chunks = chunk_shader(shader_code)

    যদি len(chunks) == 1:
        outfile.লিখ(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    অন্যথা:
        জন্য idx, chunk মধ্যে enumerate(chunks):
            outfile.লিখ(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.লিখ(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.লিখ('    static const std::string s = []{\n')
        outfile.লিখ('        std::string tmp;\n')
        outfile.লিখ(f'        tmp.reserve({len(shader_code)});\n')
        জন্য idx মধ্যে range(len(chunks)):
            outfile.লিখ(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.লিখ('        return tmp;\n')
        outfile.লিখ('    }();\n')
        outfile.লিখ('    return s;\n')
        outfile.লিখ('}\n')
        outfile.লিখ(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


def প্রধান():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=সত্য)
    parser.add_argument("--output_file", required=সত্য)
    args = parser.parse_args()

    with open(args.output_file, "w", encoding="utf-8") ইব out:
        out.লিখ("// Auto-generated shader embedding\n")
        out.লিখ("#include <string>\n\n")
        জন্য fname মধ্যে sorted(os.listdir(args.প্রবেশ_dir)):
            যদি fname.endswith(".wgsl"):
                shader_path = os.path.join(args.প্রবেশ_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                with open(shader_path, "r", encoding="utf-8") ইব f:
                    shader_code = f.পঠ()

                write_shader(shader_name, shader_code, None, out, args.প্রবেশ_dir)


যদি __name__ == "__main__":
    প্রধান()
