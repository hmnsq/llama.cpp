#ifndef PRE_WGSL_HPP
#define PRE_WGSL_HPP

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pre_wgsl {

//==============================================================
// Options
//==============================================================
struct Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
static std::string trim(const std::string & s) {
    size_t a = 0;
    пока (a < s.size() && std::isspace((unsigned char) s[a])) {
        a++;
    }
    size_t b = s.size();
    пока (b > a && std::isspace((unsigned char) s[b - 1])) {
        b--;
    }
    вернуть s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    вернуть trim(ss.str());
}

static bool isIdentChar(char c) {
    вернуть std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    пока (i > 0 && std::isspace((unsigned char) line[i - 1])) {
        i--;
    }
    вернуть i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    пока (i > 0 && std::isspace((unsigned char) line[i - 1])) {
        i--;
    }
    если (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    если (visiting.count(name)) {
        throw std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    если (it == macros.end()) {
        visiting.erase(name);
        вернуть name;
    }

    const std::string & value = it->second;
    если (value.empty()) {
        visiting.erase(name);
        вернуть "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    вернуть expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    пока (i < line.size()) {
        если (isIdentChar(line[i])) {
            size_t start = i;
            пока (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            если (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } иначе {
                result += token;
            }
        } иначе {
            result += line[i];
            i++;
        }
    }

    вернуть result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    вернуть expandMacrosRecursiveInternal(line, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
class ExprLexer {
  public:
    enum Kind { END, IDENT, NUMBER, OP, LPAREN, RPAREN };

    struct Tok {
        Kind        kind;
        std::string text;
    };

    explicit ExprLexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        если (pos >= src.size()) {
            вернуть { END, "" };
        }

        char c = src[pos];

        // number
        если (std::isdigit((unsigned char) c)) {
            size_t start = pos;
            пока (pos < src.size() && std::isdigit((unsigned char) src[pos])) {
                pos++;
            }
            вернуть { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        если (std::isalpha((unsigned char) c) || c == '_') {
            size_t start = pos;
            пока (pos < src.size() && (std::isalnum((unsigned char) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            вернуть { IDENT, std::string(src.substr(start, pos - start)) };
        }

        если (c == '(') {
            pos++;
            вернуть { LPAREN, "(" };
        }
        если (c == ')') {
            pos++;
            вернуть { RPAREN, ")" };
        }

        // multi-char operators
        static const char * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        для (auto op : two_ops) {
            если (src.substr(pos, 2) == op) {
                pos += 2;
                вернуть { OP, std::string(op) };
            }
        }

        // single-char operators
        если (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            вернуть { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        вернуть { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        пока (pos < src.size() && std::isspace((unsigned char) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
class ExprParser {
  public:
    ExprParser(std::string_view                                     expr,
               const std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(expr),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    целое parse() { вернуть parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        если (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            вернуть true;
        }
        вернуть false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        если (tok.kind == k) {
            advance();
            вернуть true;
        }
        вернуть false;
    }

    целое parseLogicalOr() {
        целое v = parseLogicalAnd();
        пока (acceptOp("||")) {
            целое rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        вернуть v;
    }

    целое parseLogicalAnd() {
        целое v = parseEquality();
        пока (acceptOp("&&")) {
            целое rhs = parseEquality();
            v       = (v && rhs);
        }
        вернуть v;
    }

    целое parseEquality() {
        целое v = parseRelational();
        для (;;) {
            если (acceptOp("==")) {
                целое rhs = parseRelational();
                v       = (v == rhs);
            } иначе если (acceptOp("!=")) {
                целое rhs = parseRelational();
                v       = (v != rhs);
            } иначе {
                break;
            }
        }
        вернуть v;
    }

    целое parseRelational() {
        целое v = parseShift();
        для (;;) {
            если (acceptOp("<")) {
                целое rhs = parseShift();
                v       = (v < rhs);
            } иначе если (acceptOp(">")) {
                целое rhs = parseShift();
                v       = (v > rhs);
            } иначе если (acceptOp("<=")) {
                целое rhs = parseShift();
                v       = (v <= rhs);
            } иначе если (acceptOp(">=")) {
                целое rhs = parseShift();
                v       = (v >= rhs);
            } иначе {
                break;
            }
        }
        вернуть v;
    }

    целое parseShift() {
        целое v = parseAdd();
        для (;;) {
            если (acceptOp("<<")) {
                целое rhs = parseAdd();
                v       = (v << rhs);
            } иначе если (acceptOp(">>")) {
                целое rhs = parseAdd();
                v       = (v >> rhs);
            } иначе {
                break;
            }
        }
        вернуть v;
    }

    целое parseAdd() {
        целое v = parseMult();
        для (;;) {
            если (acceptOp("+")) {
                целое rhs = parseMult();
                v       = (v + rhs);
            } иначе если (acceptOp("-")) {
                целое rhs = parseMult();
                v       = (v - rhs);
            } иначе {
                break;
            }
        }
        вернуть v;
    }

    целое parseMult() {
        целое v = parseUnary();
        для (;;) {
            если (acceptOp("*")) {
                целое rhs = parseUnary();
                v       = (v * rhs);
            } иначе если (acceptOp("/")) {
                целое rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } иначе если (acceptOp("%")) {
                целое rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } иначе {
                break;
            }
        }
        вернуть v;
    }

    целое parseUnary() {
        если (acceptOp("!")) {
            вернуть !parseUnary();
        }
        если (acceptOp("-")) {
            вернуть -parseUnary();
        }
        если (acceptOp("+")) {
            вернуть +parseUnary();
        }
        вернуть parsePrimary();
    }

    целое parsePrimary() {
        // '(' expr ')'
        если (acceptKind(ExprLexer::LPAREN)) {
            целое v = parse();
            если (!acceptKind(ExprLexer::RPAREN)) {
                throw std::runtime_error("missing ')'");
            }
            вернуть v;
        }

        // number
        если (tok.kind == ExprLexer::NUMBER) {
            целое v = std::stoi(tok.text);
            advance();
            вернуть v;
        }

        // defined(identifier)
        если (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            если (acceptKind(ExprLexer::LPAREN)) {
                если (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                если (!acceptKind(ExprLexer::RPAREN)) {
                    throw std::runtime_error("missing ) in defined()");
                }
                вернуть macros.count(name) ? 1 : 0;
            } иначе {
                // defined NAME
                если (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                вернуть macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        если (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            если (it == macros.end()) {
                вернуть 0;
            }
            если (it->second.empty()) {
                вернуть 1;
            }
            вернуть evalMacroExpression(name, it->second);
        }

        // unexpected
        вернуть 0;
    }

    целое evalMacroExpression(const std::string & name, const std::string & value) {
        если (visiting.count(name)) {
            throw std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        целое        v = ep.parse();
        visiting.erase(name);
        вернуть v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        если (opts_.include_path.empty()) {
            opts_.include_path = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(const std::string & filename, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::All);
        вернуть result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        вернуть result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        вернуть result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        вернуть result;
    }

  private:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    enum class DirectiveMode { All, IncludesOnly };

    struct Cond {
        bool parent_active;
        bool active;
        bool taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    void parseMacroDefinitions(const std::vector<std::string> & macro_defs) {
        для (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            если (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } иначе {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    void buildMacros(const std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        для (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        для (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            если (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } иначе {
                name  = trim(def);
                value = "";
            }

            // Add to macros map (will override global if same name)
            macros[name] = value;
            predefined.insert(name);
        }
    }

    //----------------------------------------------------------
    // Helpers
    //----------------------------------------------------------
    std::string loadFile(const std::string & fname) {
        std::ifstream f(fname);
        если (!f.is_open()) {
            throw std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        вернуть ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        если (cond.empty()) {
            вернуть true;
        }
        вернуть cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        если (include_stack.count(name)) {
            throw std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        вернуть out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        вернуть processFile(full_path, macros, predefined_macros, include_stack, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(const std::string &                            shader_code,
                              std::unordered_map<std::string, std::string> & macros,
                              const std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_stack,
                              DirectiveMode                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  out;
        std::istringstream in(shader_code);
        std::string        line;

        пока (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            если (!t.empty() && t[0] == '#') {
                пока (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    если (!std::getline(in, line)) {
                        break;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            если (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                если (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } иначе {
                если (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } иначе если (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        если (mode == DirectiveMode::All && !cond.empty()) {
            throw std::runtime_error("Unclosed #if directive");
        }

        вернуть out.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    bool handleDirective(const std::string &                            t,
                         std::stringstream &                            out,
                         std::unordered_map<std::string, std::string> & macros,
                         const std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_stack,
                         DirectiveMode                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        если (cmd == "include") {
            если (mode == DirectiveMode::All && !condActive(cond)) {
                вернуть true;
            }
            std::string file;
            iss >> file;
            если (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            вернуть true;
        }

        если (mode == DirectiveMode::IncludesOnly) {
            вернуть false;
        }

        если (cmd == "define") {
            если (!condActive(cond)) {
                вернуть true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            если (predefined_macros.count(name)) {
                вернуть true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            вернуть true;
        }

        если (cmd == "undef") {
            если (!condActive(cond)) {
                вернуть true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            если (predefined_macros.count(name)) {
                вернуть true;
            }
            macros.erase(name);
            вернуть true;
        }

        если (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            вернуть true;
        }

        если (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            вернуть true;
        }

        если (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            если (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            вернуть true;
        }

        если (cmd == "elif") {
            std::string expr = trim_value(iss);

            если (cond.empty()) {
                throw std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            если (!c.parent_active) {
                c.active = false;
                вернуть true;
            }

            если (c.taken) {
                c.active = false;
                вернуть true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            если (v) {
                c.taken = true;
            }
            вернуть true;
        }

        если (cmd == "else") {
            если (cond.empty()) {
                throw std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            если (!c.parent_active) {
                c.active = false;
                вернуть true;
            }
            если (c.taken) {
                c.active = false;
            } иначе {
                c.active = true;
                c.taken  = true;
            }
            вернуть true;
        }

        если (cmd == "endif") {
            если (cond.empty()) {
                throw std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            вернуть true;
        }

        // Unknown directive
        throw std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
