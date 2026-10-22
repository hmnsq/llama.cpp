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
    докато (a < s.size() && std::isspace((unsigned char) s[a])) {
        a++;
    }
    size_t b = s.size();
    докато (b > a && std::isspace((unsigned char) s[b - 1])) {
        b--;
    }
    върни s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    върни trim(ss.str());
}

static bool isIdentChar(char c) {
    върни std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    докато (i > 0 && std::isspace((unsigned char) line[i - 1])) {
        i--;
    }
    върни i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    докато (i > 0 && std::isspace((unsigned char) line[i - 1])) {
        i--;
    }
    ако (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ако (visiting.count(name)) {
        throw std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ако (it == macros.end()) {
        visiting.erase(name);
        върни name;
    }

    const std::string & value = it->second;
    ако (value.empty()) {
        visiting.erase(name);
        върни "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    върни expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    докато (i < line.size()) {
        ако (isIdentChar(line[i])) {
            size_t start = i;
            докато (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            ако (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } иначе {
                result += token;
            }
        } иначе {
            result += line[i];
            i++;
        }
    }

    върни result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    върни expandMacrosRecursiveInternal(line, macros, visiting);
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
        ако (pos >= src.size()) {
            върни { END, "" };
        }

        char c = src[pos];

        // number
        ако (std::isdigit((unsigned char) c)) {
            size_t start = pos;
            докато (pos < src.size() && std::isdigit((unsigned char) src[pos])) {
                pos++;
            }
            върни { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ако (std::isalpha((unsigned char) c) || c == '_') {
            size_t start = pos;
            докато (pos < src.size() && (std::isalnum((unsigned char) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            върни { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ако (c == '(') {
            pos++;
            върни { LPAREN, "(" };
        }
        ако (c == ')') {
            pos++;
            върни { RPAREN, ")" };
        }

        // multi-char operators
        static const char * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        за (auto op : two_ops) {
            ако (src.substr(pos, 2) == op) {
                pos += 2;
                върни { OP, std::string(op) };
            }
        }

        // single-char operators
        ако (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            върни { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        върни { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        докато (pos < src.size() && std::isspace((unsigned char) src[pos])) {
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

    цяло parse() { върни parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        ако (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            върни true;
        }
        върни false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        ако (tok.kind == k) {
            advance();
            върни true;
        }
        върни false;
    }

    цяло parseLogicalOr() {
        цяло v = parseLogicalAnd();
        докато (acceptOp("||")) {
            цяло rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        върни v;
    }

    цяло parseLogicalAnd() {
        цяло v = parseEquality();
        докато (acceptOp("&&")) {
            цяло rhs = parseEquality();
            v       = (v && rhs);
        }
        върни v;
    }

    цяло parseEquality() {
        цяло v = parseRelational();
        за (;;) {
            ако (acceptOp("==")) {
                цяло rhs = parseRelational();
                v       = (v == rhs);
            } иначе ако (acceptOp("!=")) {
                цяло rhs = parseRelational();
                v       = (v != rhs);
            } иначе {
                break;
            }
        }
        върни v;
    }

    цяло parseRelational() {
        цяло v = parseShift();
        за (;;) {
            ако (acceptOp("<")) {
                цяло rhs = parseShift();
                v       = (v < rhs);
            } иначе ако (acceptOp(">")) {
                цяло rhs = parseShift();
                v       = (v > rhs);
            } иначе ако (acceptOp("<=")) {
                цяло rhs = parseShift();
                v       = (v <= rhs);
            } иначе ако (acceptOp(">=")) {
                цяло rhs = parseShift();
                v       = (v >= rhs);
            } иначе {
                break;
            }
        }
        върни v;
    }

    цяло parseShift() {
        цяло v = parseAdd();
        за (;;) {
            ако (acceptOp("<<")) {
                цяло rhs = parseAdd();
                v       = (v << rhs);
            } иначе ако (acceptOp(">>")) {
                цяло rhs = parseAdd();
                v       = (v >> rhs);
            } иначе {
                break;
            }
        }
        върни v;
    }

    цяло parseAdd() {
        цяло v = parseMult();
        за (;;) {
            ако (acceptOp("+")) {
                цяло rhs = parseMult();
                v       = (v + rhs);
            } иначе ако (acceptOp("-")) {
                цяло rhs = parseMult();
                v       = (v - rhs);
            } иначе {
                break;
            }
        }
        върни v;
    }

    цяло parseMult() {
        цяло v = parseUnary();
        за (;;) {
            ако (acceptOp("*")) {
                цяло rhs = parseUnary();
                v       = (v * rhs);
            } иначе ако (acceptOp("/")) {
                цяло rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } иначе ако (acceptOp("%")) {
                цяло rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } иначе {
                break;
            }
        }
        върни v;
    }

    цяло parseUnary() {
        ако (acceptOp("!")) {
            върни !parseUnary();
        }
        ако (acceptOp("-")) {
            върни -parseUnary();
        }
        ако (acceptOp("+")) {
            върни +parseUnary();
        }
        върни parsePrimary();
    }

    цяло parsePrimary() {
        // '(' expr ')'
        ако (acceptKind(ExprLexer::LPAREN)) {
            цяло v = parse();
            ако (!acceptKind(ExprLexer::RPAREN)) {
                throw std::runtime_error("missing ')'");
            }
            върни v;
        }

        // number
        ако (tok.kind == ExprLexer::NUMBER) {
            цяло v = std::stoi(tok.text);
            advance();
            върни v;
        }

        // defined(identifier)
        ако (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            ако (acceptKind(ExprLexer::LPAREN)) {
                ако (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ако (!acceptKind(ExprLexer::RPAREN)) {
                    throw std::runtime_error("missing ) in defined()");
                }
                върни macros.count(name) ? 1 : 0;
            } иначе {
                // defined NAME
                ако (tok.kind != ExprLexer::IDENT) {
                    throw std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                върни macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ако (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ако (it == macros.end()) {
                върни 0;
            }
            ако (it->second.empty()) {
                върни 1;
            }
            върни evalMacroExpression(name, it->second);
        }

        // unexpected
        върни 0;
    }

    цяло evalMacroExpression(const std::string & name, const std::string & value) {
        ако (visiting.count(name)) {
            throw std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        цяло        v = ep.parse();
        visiting.erase(name);
        върни v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ако (opts_.include_path.empty()) {
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
        върни result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        върни result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        върни result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        върни result;
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
        за (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ако (eq_pos != std::string::npos) {
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

        за (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        за (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ако (eq_pos != std::string::npos) {
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
        ако (!f.is_open()) {
            throw std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        върни ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        ако (cond.empty()) {
            върни true;
        }
        върни cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        ако (include_stack.count(name)) {
            throw std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        върни out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        върни processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        докато (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            ако (!t.empty() && t[0] == '#') {
                докато (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ако (!std::getline(in, line)) {
                        break;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            ако (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                ако (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } иначе {
                ако (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } иначе ако (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ако (mode == DirectiveMode::All && !cond.empty()) {
            throw std::runtime_error("Unclosed #if directive");
        }

        върни out.str();
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

        ако (cmd == "include") {
            ако (mode == DirectiveMode::All && !condActive(cond)) {
                върни true;
            }
            std::string file;
            iss >> file;
            ако (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            върни true;
        }

        ако (mode == DirectiveMode::IncludesOnly) {
            върни false;
        }

        ако (cmd == "define") {
            ако (!condActive(cond)) {
                върни true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ако (predefined_macros.count(name)) {
                върни true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            върни true;
        }

        ако (cmd == "undef") {
            ако (!condActive(cond)) {
                върни true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ако (predefined_macros.count(name)) {
                върни true;
            }
            macros.erase(name);
            върни true;
        }

        ако (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            върни true;
        }

        ако (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            върни true;
        }

        ако (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            ако (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            върни true;
        }

        ако (cmd == "elif") {
            std::string expr = trim_value(iss);

            ако (cond.empty()) {
                throw std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ако (!c.parent_active) {
                c.active = false;
                върни true;
            }

            ако (c.taken) {
                c.active = false;
                върни true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            ако (v) {
                c.taken = true;
            }
            върни true;
        }

        ако (cmd == "else") {
            ако (cond.empty()) {
                throw std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ако (!c.parent_active) {
                c.active = false;
                върни true;
            }
            ако (c.taken) {
                c.active = false;
            } иначе {
                c.active = true;
                c.taken  = true;
            }
            върни true;
        }

        ако (cmd == "endif") {
            ако (cond.empty()) {
                throw std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            върни true;
        }

        // Unknown directive
        throw std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
