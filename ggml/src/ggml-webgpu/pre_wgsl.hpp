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
    trong_khi (a < s.size() && std::isspace((unsigned char) s[a])) {
        a++;
    }
    size_t b = s.size();
    trong_khi (b > a && std::isspace((unsigned char) s[b - 1])) {
        b--;
    }
    trả_về s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    trả_về trim(ss.str());
}

static bool isIdentChar(char c) {
    trả_về std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    trong_khi (i > 0 && std::isspace((unsigned char) line[i - 1])) {
        i--;
    }
    trả_về i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    trong_khi (i > 0 && std::isspace((unsigned char) line[i - 1])) {
        i--;
    }
    nếu (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    nếu (visiting.count(name)) {
        throw std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    nếu (it == macros.end()) {
        visiting.erase(name);
        trả_về name;
    }

    const std::string & value = it->second;
    nếu (value.empty()) {
        visiting.erase(name);
        trả_về "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    trả_về expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    trong_khi (i < line.size()) {
        nếu (isIdentChar(line[i])) {
            size_t start = i;
            trong_khi (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            nếu (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } không_thì {
                result += token;
            }
        } không_thì {
            result += line[i];
            i++;
        }
    }

    trả_về result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    trả_về expandMacrosRecursiveInternal(line, macros, visiting);
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
        nếu (pos >= src.size()) {
            trả_về { END, "" };
        }

        char c = src[pos];

        // number
        nếu (std::isdigit((unsigned char) c)) {
            size_t start = pos;
            trong_khi (pos < src.size() && std::isdigit((unsigned char) src[pos])) {
                pos++;
            }
            trả_về { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        nếu (std::isalpha((unsigned char) c) || c == '_') {
            size_t start = pos;
            trong_khi (pos < src.size() && (std::isalnum((unsigned char) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            trả_về { IDENT, std::string(src.substr(start, pos - start)) };
        }

        nếu (c == '(') {
            pos++;
            trả_về { LPAREN, "(" };
        }
        nếu (c == ')') {
            pos++;
            trả_về { RPAREN, ")" };
        }

        // multi-char operators
        static const char * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        cho (auto op : two_ops) {
            nếu (src.substr(pos, 2) == op) {
                pos += 2;
                trả_về { OP, std::string(op) };
            }
        }

        // single-char operators
        nếu (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            trả_về { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        trả_về { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        trong_khi (pos < src.size() && std::isspace((unsigned char) src[pos])) {
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

    số_nguyên parse() { trả_về parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        nếu (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            trả_về true;
        }
        trả_về false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        nếu (tok.kind == k) {
            advance();
            trả_về true;
        }
        trả_về false;
    }

    số_nguyên parseLogicalOr() {
        số_nguyên v = parseLogicalAnd();
        trong_khi (acceptOp("||")) {
            số_nguyên rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        trả_về v;
    }

    số_nguyên parseLogicalAnd() {
        số_nguyên v = parseEquality();
        trong_khi (acceptOp("&&")) {
            số_nguyên rhs = parseEquality();
            v       = (v && rhs);
        }
        trả_về v;
    }

    số_nguyên parseEquality() {
        số_nguyên v = parseRelational();
        cho (;;) {
            nếu (acceptOp("==")) {
                số_nguyên rhs = parseRelational();
                v       = (v == rhs);
            } không_thì nếu (acceptOp("!=")) {
                số_nguyên rhs = parseRelational();
                v       = (v != rhs);
            } không_thì {
                break;
            }
        }
        trả_về v;
    }

    số_nguyên parseRelational() {
        số_nguyên v = parseShift();
        cho (;;) {
            nếu (acceptOp("<")) {
                số_nguyên rhs = parseShift();
                v       = (v < rhs);
            } không_thì nếu (acceptOp(">")) {
                số_nguyên rhs = parseShift();
                v       = (v > rhs);
            } không_thì nếu (acceptOp("<=")) {
                số_nguyên rhs = parseShift();
                v       = (v <= rhs);
            } không_thì nếu (acceptOp(">=")) {
                số_nguyên rhs = parseShift();
                v       = (v >= rhs);
            } không_thì {
                break;
            }
        }
        trả_về v;
    }

    số_nguyên parseShift() {
        số_nguyên v = parseAdd();
        cho (;;) {
            nếu (acceptOp("<<")) {
                số_nguyên rhs = parseAdd();
                v       = (v << rhs);
            } không_thì nếu (acceptOp(">>")) {
                số_nguyên rhs = parseAdd();
                v       = (v >> rhs);
            } không_thì {
                break;
            }
        }
        trả_về v;
    }

    số_nguyên parseAdd() {
        số_nguyên v = parseMult();
        cho (;;) {
            nếu (acceptOp("+")) {
                số_nguyên rhs = parseMult();
                v       = (v + rhs);
            } không_thì nếu (acceptOp("-")) {
                số_nguyên rhs = parseMult();
                v       = (v - rhs);
            } không_thì {
                break;
            }
        }
        trả_về v;
    }

    số_nguyên parseMult() {
        số_nguyên v = parseUnary();
        cho (;;) {
            nếu (acceptOp("*")) {
                số_nguyên rhs = parseUnary();
                v       = (v * rhs);
            } không_thì nếu (acceptOp("/")) {
                số_nguyên rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } không_thì nếu (acceptOp("%")) {
                số_nguyên rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } không_thì {
                break;
            }
        }
        trả_về v;
    }

    số_nguyên parseUnary() {
        nếu (acceptOp("!")) {
            trả_về !parseUnary();
        }
        nếu (acceptOp("-")) {
            trả_về -parseUnary();
        }
        nếu (acceptOp("+")) {
            trả_về +parseUnary();
        }
        trả_về parsePrimary();
    }

    số_nguyên parsePrimary() {
        // '(' expr ')'
        nếu (acceptKind(ExprLexer::LPAREN)) {
            số_nguyên v = parse();
            nếu (!acceptKind(ExprLexer::RPAREN)) {
                throw std::runtime_error("missing ')'");
            }
            trả_về v;
        }

        // number
        nếu (tok.kind == ExprLexer::NUMBER) {
            số_nguyên v = std::stoi(tok.text);
            advance();
            trả_về v;
        }

        // defined(identifier)
        nếu (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            nếu (acceptKind(ExprLexer::LPAREN)) {
                nếu (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                nếu (!acceptKind(ExprLexer::RPAREN)) {
                    throw std::runtime_error("missing ) in defined()");
                }
                trả_về macros.count(name) ? 1 : 0;
            } không_thì {
                // defined NAME
                nếu (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                trả_về macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        nếu (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            nếu (it == macros.end()) {
                trả_về 0;
            }
            nếu (it->second.empty()) {
                trả_về 1;
            }
            trả_về evalMacroExpression(name, it->second);
        }

        // unexpected
        trả_về 0;
    }

    số_nguyên evalMacroExpression(const std::string & name, const std::string & value) {
        nếu (visiting.count(name)) {
            throw std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        số_nguyên        v = ep.parse();
        visiting.erase(name);
        trả_về v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        nếu (opts_.include_path.empty()) {
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
        trả_về result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        trả_về result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        trả_về result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        trả_về result;
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
        cho (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            nếu (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } không_thì {
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

        cho (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        cho (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            nếu (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } không_thì {
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
        nếu (!f.is_open()) {
            throw std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        trả_về ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        nếu (cond.empty()) {
            trả_về true;
        }
        trả_về cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        nếu (include_stack.count(name)) {
            throw std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        trả_về out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        trả_về processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        trong_khi (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            nếu (!t.empty() && t[0] == '#') {
                trong_khi (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    nếu (!std::getline(in, line)) {
                        break;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            nếu (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                nếu (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } không_thì {
                nếu (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } không_thì nếu (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        nếu (mode == DirectiveMode::All && !cond.empty()) {
            throw std::runtime_error("Unclosed #if directive");
        }

        trả_về out.str();
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

        nếu (cmd == "include") {
            nếu (mode == DirectiveMode::All && !condActive(cond)) {
                trả_về true;
            }
            std::string file;
            iss >> file;
            nếu (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            trả_về true;
        }

        nếu (mode == DirectiveMode::IncludesOnly) {
            trả_về false;
        }

        nếu (cmd == "define") {
            nếu (!condActive(cond)) {
                trả_về true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            nếu (predefined_macros.count(name)) {
                trả_về true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            trả_về true;
        }

        nếu (cmd == "undef") {
            nếu (!condActive(cond)) {
                trả_về true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            nếu (predefined_macros.count(name)) {
                trả_về true;
            }
            macros.erase(name);
            trả_về true;
        }

        nếu (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            trả_về true;
        }

        nếu (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            trả_về true;
        }

        nếu (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            nếu (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            trả_về true;
        }

        nếu (cmd == "elif") {
            std::string expr = trim_value(iss);

            nếu (cond.empty()) {
                throw std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            nếu (!c.parent_active) {
                c.active = false;
                trả_về true;
            }

            nếu (c.taken) {
                c.active = false;
                trả_về true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            nếu (v) {
                c.taken = true;
            }
            trả_về true;
        }

        nếu (cmd == "else") {
            nếu (cond.empty()) {
                throw std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            nếu (!c.parent_active) {
                c.active = false;
                trả_về true;
            }
            nếu (c.taken) {
                c.active = false;
            } không_thì {
                c.active = true;
                c.taken  = true;
            }
            trả_về true;
        }

        nếu (cmd == "endif") {
            nếu (cond.empty()) {
                throw std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            trả_về true;
        }

        // Unknown directive
        throw std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
