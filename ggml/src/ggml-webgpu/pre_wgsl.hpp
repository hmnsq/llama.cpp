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
    当 (a < s.size() && std::isspace((unsigned char) s[a])) {
        a++;
    }
    size_t b = s.size();
    当 (b > a && std::isspace((unsigned char) s[b - 1])) {
        b--;
    }
    返回 s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    返回 trim(ss.str());
}

static bool isIdentChar(char c) {
    返回 std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    当 (i > 0 && std::isspace((unsigned char) line[i - 1])) {
        i--;
    }
    返回 i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    当 (i > 0 && std::isspace((unsigned char) line[i - 1])) {
        i--;
    }
    如果 (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    如果 (visiting.count(name)) {
        throw std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    如果 (it == macros.end()) {
        visiting.erase(name);
        返回 name;
    }

    const std::string & value = it->second;
    如果 (value.empty()) {
        visiting.erase(name);
        返回 "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    返回 expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    当 (i < line.size()) {
        如果 (isIdentChar(line[i])) {
            size_t start = i;
            当 (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            如果 (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } 否则 {
                result += token;
            }
        } 否则 {
            result += line[i];
            i++;
        }
    }

    返回 result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    返回 expandMacrosRecursiveInternal(line, macros, visiting);
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
        如果 (pos >= src.size()) {
            返回 { END, "" };
        }

        char c = src[pos];

        // number
        如果 (std::isdigit((unsigned char) c)) {
            size_t start = pos;
            当 (pos < src.size() && std::isdigit((unsigned char) src[pos])) {
                pos++;
            }
            返回 { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        如果 (std::isalpha((unsigned char) c) || c == '_') {
            size_t start = pos;
            当 (pos < src.size() && (std::isalnum((unsigned char) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            返回 { IDENT, std::string(src.substr(start, pos - start)) };
        }

        如果 (c == '(') {
            pos++;
            返回 { LPAREN, "(" };
        }
        如果 (c == ')') {
            pos++;
            返回 { RPAREN, ")" };
        }

        // multi-char operators
        static const char * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        为 (auto op : two_ops) {
            如果 (src.substr(pos, 2) == op) {
                pos += 2;
                返回 { OP, std::string(op) };
            }
        }

        // single-char operators
        如果 (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            返回 { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        返回 { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        当 (pos < src.size() && std::isspace((unsigned char) src[pos])) {
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

    整数 parse() { 返回 parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        如果 (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            返回 true;
        }
        返回 false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        如果 (tok.kind == k) {
            advance();
            返回 true;
        }
        返回 false;
    }

    整数 parseLogicalOr() {
        整数 v = parseLogicalAnd();
        当 (acceptOp("||")) {
            整数 rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        返回 v;
    }

    整数 parseLogicalAnd() {
        整数 v = parseEquality();
        当 (acceptOp("&&")) {
            整数 rhs = parseEquality();
            v       = (v && rhs);
        }
        返回 v;
    }

    整数 parseEquality() {
        整数 v = parseRelational();
        为 (;;) {
            如果 (acceptOp("==")) {
                整数 rhs = parseRelational();
                v       = (v == rhs);
            } 否则 如果 (acceptOp("!=")) {
                整数 rhs = parseRelational();
                v       = (v != rhs);
            } 否则 {
                break;
            }
        }
        返回 v;
    }

    整数 parseRelational() {
        整数 v = parseShift();
        为 (;;) {
            如果 (acceptOp("<")) {
                整数 rhs = parseShift();
                v       = (v < rhs);
            } 否则 如果 (acceptOp(">")) {
                整数 rhs = parseShift();
                v       = (v > rhs);
            } 否则 如果 (acceptOp("<=")) {
                整数 rhs = parseShift();
                v       = (v <= rhs);
            } 否则 如果 (acceptOp(">=")) {
                整数 rhs = parseShift();
                v       = (v >= rhs);
            } 否则 {
                break;
            }
        }
        返回 v;
    }

    整数 parseShift() {
        整数 v = parseAdd();
        为 (;;) {
            如果 (acceptOp("<<")) {
                整数 rhs = parseAdd();
                v       = (v << rhs);
            } 否则 如果 (acceptOp(">>")) {
                整数 rhs = parseAdd();
                v       = (v >> rhs);
            } 否则 {
                break;
            }
        }
        返回 v;
    }

    整数 parseAdd() {
        整数 v = parseMult();
        为 (;;) {
            如果 (acceptOp("+")) {
                整数 rhs = parseMult();
                v       = (v + rhs);
            } 否则 如果 (acceptOp("-")) {
                整数 rhs = parseMult();
                v       = (v - rhs);
            } 否则 {
                break;
            }
        }
        返回 v;
    }

    整数 parseMult() {
        整数 v = parseUnary();
        为 (;;) {
            如果 (acceptOp("*")) {
                整数 rhs = parseUnary();
                v       = (v * rhs);
            } 否则 如果 (acceptOp("/")) {
                整数 rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } 否则 如果 (acceptOp("%")) {
                整数 rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } 否则 {
                break;
            }
        }
        返回 v;
    }

    整数 parseUnary() {
        如果 (acceptOp("!")) {
            返回 !parseUnary();
        }
        如果 (acceptOp("-")) {
            返回 -parseUnary();
        }
        如果 (acceptOp("+")) {
            返回 +parseUnary();
        }
        返回 parsePrimary();
    }

    整数 parsePrimary() {
        // '(' expr ')'
        如果 (acceptKind(ExprLexer::LPAREN)) {
            整数 v = parse();
            如果 (!acceptKind(ExprLexer::RPAREN)) {
                throw std::runtime_error("missing ')'");
            }
            返回 v;
        }

        // number
        如果 (tok.kind == ExprLexer::NUMBER) {
            整数 v = std::stoi(tok.text);
            advance();
            返回 v;
        }

        // defined(identifier)
        如果 (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            如果 (acceptKind(ExprLexer::LPAREN)) {
                如果 (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                如果 (!acceptKind(ExprLexer::RPAREN)) {
                    throw std::runtime_error("missing ) in defined()");
                }
                返回 macros.count(name) ? 1 : 0;
            } 否则 {
                // defined NAME
                如果 (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                返回 macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        如果 (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            如果 (it == macros.end()) {
                返回 0;
            }
            如果 (it->second.empty()) {
                返回 1;
            }
            返回 evalMacroExpression(name, it->second);
        }

        // unexpected
        返回 0;
    }

    整数 evalMacroExpression(const std::string & name, const std::string & value) {
        如果 (visiting.count(name)) {
            throw std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        整数        v = ep.parse();
        visiting.erase(name);
        返回 v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        如果 (opts_.include_path.empty()) {
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
        返回 result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        返回 result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        返回 result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        返回 result;
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
        为 (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            如果 (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } 否则 {
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

        为 (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        为 (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            如果 (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } 否则 {
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
        如果 (!f.is_open()) {
            throw std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        返回 ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        如果 (cond.empty()) {
            返回 true;
        }
        返回 cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        如果 (include_stack.count(name)) {
            throw std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        返回 out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        返回 processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        当 (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            如果 (!t.empty() && t[0] == '#') {
                当 (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    如果 (!std::getline(in, line)) {
                        break;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            如果 (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                如果 (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } 否则 {
                如果 (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } 否则 如果 (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        如果 (mode == DirectiveMode::All && !cond.empty()) {
            throw std::runtime_error("Unclosed #if directive");
        }

        返回 out.str();
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

        如果 (cmd == "include") {
            如果 (mode == DirectiveMode::All && !condActive(cond)) {
                返回 true;
            }
            std::string file;
            iss >> file;
            如果 (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            返回 true;
        }

        如果 (mode == DirectiveMode::IncludesOnly) {
            返回 false;
        }

        如果 (cmd == "define") {
            如果 (!condActive(cond)) {
                返回 true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            如果 (predefined_macros.count(name)) {
                返回 true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            返回 true;
        }

        如果 (cmd == "undef") {
            如果 (!condActive(cond)) {
                返回 true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            如果 (predefined_macros.count(name)) {
                返回 true;
            }
            macros.erase(name);
            返回 true;
        }

        如果 (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            返回 true;
        }

        如果 (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            返回 true;
        }

        如果 (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            如果 (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            返回 true;
        }

        如果 (cmd == "elif") {
            std::string expr = trim_value(iss);

            如果 (cond.empty()) {
                throw std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            如果 (!c.parent_active) {
                c.active = false;
                返回 true;
            }

            如果 (c.taken) {
                c.active = false;
                返回 true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            如果 (v) {
                c.taken = true;
            }
            返回 true;
        }

        如果 (cmd == "else") {
            如果 (cond.empty()) {
                throw std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            如果 (!c.parent_active) {
                c.active = false;
                返回 true;
            }
            如果 (c.taken) {
                c.active = false;
            } 否则 {
                c.active = true;
                c.taken  = true;
            }
            返回 true;
        }

        如果 (cmd == "endif") {
            如果 (cond.empty()) {
                throw std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            返回 true;
        }

        // Unknown directive
        throw std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
