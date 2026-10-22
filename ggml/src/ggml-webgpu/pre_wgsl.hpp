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
    동안 (a < s.size() && std::isspace((unsigned char) s[a])) {
        a++;
    }
    size_t b = s.size();
    동안 (b > a && std::isspace((unsigned char) s[b - 1])) {
        b--;
    }
    반환 s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    반환 trim(ss.str());
}

static bool isIdentChar(char c) {
    반환 std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    동안 (i > 0 && std::isspace((unsigned char) line[i - 1])) {
        i--;
    }
    반환 i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    동안 (i > 0 && std::isspace((unsigned char) line[i - 1])) {
        i--;
    }
    만약 (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    만약 (visiting.count(name)) {
        throw std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    만약 (it == macros.end()) {
        visiting.erase(name);
        반환 name;
    }

    const std::string & value = it->second;
    만약 (value.empty()) {
        visiting.erase(name);
        반환 "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    반환 expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    동안 (i < line.size()) {
        만약 (isIdentChar(line[i])) {
            size_t start = i;
            동안 (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            만약 (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } 아니면 {
                result += token;
            }
        } 아니면 {
            result += line[i];
            i++;
        }
    }

    반환 result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    반환 expandMacrosRecursiveInternal(line, macros, visiting);
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
        만약 (pos >= src.size()) {
            반환 { END, "" };
        }

        char c = src[pos];

        // number
        만약 (std::isdigit((unsigned char) c)) {
            size_t start = pos;
            동안 (pos < src.size() && std::isdigit((unsigned char) src[pos])) {
                pos++;
            }
            반환 { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        만약 (std::isalpha((unsigned char) c) || c == '_') {
            size_t start = pos;
            동안 (pos < src.size() && (std::isalnum((unsigned char) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            반환 { IDENT, std::string(src.substr(start, pos - start)) };
        }

        만약 (c == '(') {
            pos++;
            반환 { LPAREN, "(" };
        }
        만약 (c == ')') {
            pos++;
            반환 { RPAREN, ")" };
        }

        // multi-char operators
        static const char * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        위해 (auto op : two_ops) {
            만약 (src.substr(pos, 2) == op) {
                pos += 2;
                반환 { OP, std::string(op) };
            }
        }

        // single-char operators
        만약 (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            반환 { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        반환 { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        동안 (pos < src.size() && std::isspace((unsigned char) src[pos])) {
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

    정수 parse() { 반환 parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        만약 (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            반환 true;
        }
        반환 false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        만약 (tok.kind == k) {
            advance();
            반환 true;
        }
        반환 false;
    }

    정수 parseLogicalOr() {
        정수 v = parseLogicalAnd();
        동안 (acceptOp("||")) {
            정수 rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        반환 v;
    }

    정수 parseLogicalAnd() {
        정수 v = parseEquality();
        동안 (acceptOp("&&")) {
            정수 rhs = parseEquality();
            v       = (v && rhs);
        }
        반환 v;
    }

    정수 parseEquality() {
        정수 v = parseRelational();
        위해 (;;) {
            만약 (acceptOp("==")) {
                정수 rhs = parseRelational();
                v       = (v == rhs);
            } 아니면 만약 (acceptOp("!=")) {
                정수 rhs = parseRelational();
                v       = (v != rhs);
            } 아니면 {
                break;
            }
        }
        반환 v;
    }

    정수 parseRelational() {
        정수 v = parseShift();
        위해 (;;) {
            만약 (acceptOp("<")) {
                정수 rhs = parseShift();
                v       = (v < rhs);
            } 아니면 만약 (acceptOp(">")) {
                정수 rhs = parseShift();
                v       = (v > rhs);
            } 아니면 만약 (acceptOp("<=")) {
                정수 rhs = parseShift();
                v       = (v <= rhs);
            } 아니면 만약 (acceptOp(">=")) {
                정수 rhs = parseShift();
                v       = (v >= rhs);
            } 아니면 {
                break;
            }
        }
        반환 v;
    }

    정수 parseShift() {
        정수 v = parseAdd();
        위해 (;;) {
            만약 (acceptOp("<<")) {
                정수 rhs = parseAdd();
                v       = (v << rhs);
            } 아니면 만약 (acceptOp(">>")) {
                정수 rhs = parseAdd();
                v       = (v >> rhs);
            } 아니면 {
                break;
            }
        }
        반환 v;
    }

    정수 parseAdd() {
        정수 v = parseMult();
        위해 (;;) {
            만약 (acceptOp("+")) {
                정수 rhs = parseMult();
                v       = (v + rhs);
            } 아니면 만약 (acceptOp("-")) {
                정수 rhs = parseMult();
                v       = (v - rhs);
            } 아니면 {
                break;
            }
        }
        반환 v;
    }

    정수 parseMult() {
        정수 v = parseUnary();
        위해 (;;) {
            만약 (acceptOp("*")) {
                정수 rhs = parseUnary();
                v       = (v * rhs);
            } 아니면 만약 (acceptOp("/")) {
                정수 rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } 아니면 만약 (acceptOp("%")) {
                정수 rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } 아니면 {
                break;
            }
        }
        반환 v;
    }

    정수 parseUnary() {
        만약 (acceptOp("!")) {
            반환 !parseUnary();
        }
        만약 (acceptOp("-")) {
            반환 -parseUnary();
        }
        만약 (acceptOp("+")) {
            반환 +parseUnary();
        }
        반환 parsePrimary();
    }

    정수 parsePrimary() {
        // '(' expr ')'
        만약 (acceptKind(ExprLexer::LPAREN)) {
            정수 v = parse();
            만약 (!acceptKind(ExprLexer::RPAREN)) {
                throw std::runtime_error("missing ')'");
            }
            반환 v;
        }

        // number
        만약 (tok.kind == ExprLexer::NUMBER) {
            정수 v = std::stoi(tok.text);
            advance();
            반환 v;
        }

        // defined(identifier)
        만약 (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            만약 (acceptKind(ExprLexer::LPAREN)) {
                만약 (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                만약 (!acceptKind(ExprLexer::RPAREN)) {
                    throw std::runtime_error("missing ) in defined()");
                }
                반환 macros.count(name) ? 1 : 0;
            } 아니면 {
                // defined NAME
                만약 (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                반환 macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        만약 (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            만약 (it == macros.end()) {
                반환 0;
            }
            만약 (it->second.empty()) {
                반환 1;
            }
            반환 evalMacroExpression(name, it->second);
        }

        // unexpected
        반환 0;
    }

    정수 evalMacroExpression(const std::string & name, const std::string & value) {
        만약 (visiting.count(name)) {
            throw std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        정수        v = ep.parse();
        visiting.erase(name);
        반환 v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        만약 (opts_.include_path.empty()) {
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
        반환 result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        반환 result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        반환 result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        반환 result;
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
        위해 (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            만약 (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } 아니면 {
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

        위해 (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        위해 (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            만약 (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } 아니면 {
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
        만약 (!f.is_open()) {
            throw std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        반환 ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        만약 (cond.empty()) {
            반환 true;
        }
        반환 cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        만약 (include_stack.count(name)) {
            throw std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        반환 out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        반환 processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        동안 (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            만약 (!t.empty() && t[0] == '#') {
                동안 (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    만약 (!std::getline(in, line)) {
                        break;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            만약 (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                만약 (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } 아니면 {
                만약 (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } 아니면 만약 (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        만약 (mode == DirectiveMode::All && !cond.empty()) {
            throw std::runtime_error("Unclosed #if directive");
        }

        반환 out.str();
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

        만약 (cmd == "include") {
            만약 (mode == DirectiveMode::All && !condActive(cond)) {
                반환 true;
            }
            std::string file;
            iss >> file;
            만약 (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            반환 true;
        }

        만약 (mode == DirectiveMode::IncludesOnly) {
            반환 false;
        }

        만약 (cmd == "define") {
            만약 (!condActive(cond)) {
                반환 true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            만약 (predefined_macros.count(name)) {
                반환 true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            반환 true;
        }

        만약 (cmd == "undef") {
            만약 (!condActive(cond)) {
                반환 true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            만약 (predefined_macros.count(name)) {
                반환 true;
            }
            macros.erase(name);
            반환 true;
        }

        만약 (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            반환 true;
        }

        만약 (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            반환 true;
        }

        만약 (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            만약 (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            반환 true;
        }

        만약 (cmd == "elif") {
            std::string expr = trim_value(iss);

            만약 (cond.empty()) {
                throw std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            만약 (!c.parent_active) {
                c.active = false;
                반환 true;
            }

            만약 (c.taken) {
                c.active = false;
                반환 true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            만약 (v) {
                c.taken = true;
            }
            반환 true;
        }

        만약 (cmd == "else") {
            만약 (cond.empty()) {
                throw std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            만약 (!c.parent_active) {
                c.active = false;
                반환 true;
            }
            만약 (c.taken) {
                c.active = false;
            } 아니면 {
                c.active = true;
                c.taken  = true;
            }
            반환 true;
        }

        만약 (cmd == "endif") {
            만약 (cond.empty()) {
                throw std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            반환 true;
        }

        // Unknown directive
        throw std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
