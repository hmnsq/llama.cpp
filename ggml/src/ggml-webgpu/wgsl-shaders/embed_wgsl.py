આયાત os
આયાત re
આયાત argparse


def expand_includes(shader, આગત_dir):
    """
    Replace #include "file" lines in the text with the contents of that file.
    Searches for files relative to input_dir.
    """
    include_pattern = re.compile(r'^\s*#include\s+"([^"]+)"\s*$', re.MULTILINE)

    def replacer(match):
        fname = match.group(1)
        file_path = os.path.join(આગત_dir, fname)
        જો નહીં os.path.exists(file_path):
            ઉઠાવો FileNotFoundError(f"Included file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") ઇવ f:
            included_code = f.પઠ()
        # Recursively expand includes inside the included file
        પાછા expand_includes(included_code, આગત_dir)

    પાછા include_pattern.sub(replacer, shader)


def chunk_shader(shader_code, max_chunk_len=60000):
    """Split shader_code into safe raw-string sized chunks."""
    પાછા [shader_code[i : i + max_chunk_len] માટે i માં range(0, len(shader_code), max_chunk_len)]


def raw_delim(shader_code):
    """Pick a raw-string delimiter that does not appear in the shader."""
    delim = "wgsl"
    જ્યારે f"){delim}\"" માં shader_code:
        delim += "_x"
    પાછા delim


def write_shader(shader_name, shader_code, output_dir, outfile, આગત_dir):
    shader_code = expand_includes(shader_code, આગત_dir)

    જો output_dir:
        wgsl_filename = os.path.join(output_dir, f"{shader_name}.wgsl")
        with open(wgsl_filename, "w", encoding="utf-8") ઇવ f_out:
            f_out.લિખ(shader_code)

    delim = raw_delim(shader_code)
    chunks = chunk_shader(shader_code)

    જો len(chunks) == 1:
        outfile.લિખ(f'const char* wgsl_{shader_name} = R"{delim}({shader_code}){delim}";\n\n')
    નહીંતર:
        માટે idx, chunk માં enumerate(chunks):
            outfile.લિખ(f'static const char wgsl_{shader_name}_part{idx}[] = R"{delim}({chunk}){delim}";\n\n')
        outfile.લિખ(f'static const std::string& wgsl_{shader_name}_str() {{\n')
        outfile.લિખ('    static const std::string s = []{\n')
        outfile.લિખ('        std::string tmp;\n')
        outfile.લિખ(f'        tmp.reserve({len(shader_code)});\n')
        માટે idx માં range(len(chunks)):
            outfile.લિખ(f'        tmp.append(wgsl_{shader_name}_part{idx});\n')
        outfile.લિખ('        return tmp;\n')
        outfile.લિખ('    }();\n')
        outfile.લિખ('    return s;\n')
        outfile.લિખ('}\n')
        outfile.લિખ(f'const char* wgsl_{shader_name} = wgsl_{shader_name}_str().c_str();\n\n')


def મુખ્ય():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=સાચું)
    parser.add_argument("--output_file", required=સાચું)
    args = parser.parse_args()

    with open(args.output_file, "w", encoding="utf-8") ઇવ out:
        out.લિખ("// Auto-generated shader embedding\n")
        out.લિખ("#include <string>\n\n")
        માટે fname માં sorted(os.listdir(args.આગત_dir)):
            જો fname.endswith(".wgsl"):
                shader_path = os.path.join(args.આગત_dir, fname)
                shader_name = fname.replace(".wgsl", "")

                with open(shader_path, "r", encoding="utf-8") ઇવ f:
                    shader_code = f.પઠ()

                write_shader(shader_name, shader_code, None, out, args.આગત_dir)


જો __name__ == "__main__":
    મુખ્ય()
