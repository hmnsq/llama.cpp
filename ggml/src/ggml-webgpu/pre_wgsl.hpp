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
sa_nracanaa Options {
    std::string              include_path = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
static std::string trim(const std::string & s) {
    size_t a = 0;
    ขณะ (a < s.size() && std::isspace((unsigned var_na) s[a])) {
        a++;
    }
    size_t b = s.size();
    ขณะ (b > a && std::isspace((unsigned var_na) s[b - 1])) {
        b--;
    }
    คืน s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    คืน trim(ss.str());
}

static bool isIdentChar(var_na c) {
    คืน std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    ขณะ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    คืน i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    ขณะ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ถ้า (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ถ้า (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ถ้า (it == macros.end()) {
        visiting.erase(name);
        คืน name;
    }

    const std::string & value = it->second;
    ถ้า (value.empty()) {
        visiting.erase(name);
        คืน "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    คืน expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    ขณะ (i < line.size()) {
        ถ้า (isIdentChar(line[i])) {
            size_t start = i;
            ขณะ (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            ถ้า (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } มิฉะนั้น {
                result += token;
            }
        } มิฉะนั้น {
            result += line[i];
            i++;
        }
    }

    คืน result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    คืน expandMacrosRecursiveInternal(line, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
class ExprLexer {
  public:
    ga_nanaa Kind { END, IDENT, NUMBER, OP, LPAREN, RPAREN };

    sa_nracanaa Tok {
        Kind        kind;
        std::string text;
    };

    explicit ExprLexer(std::string_view sv) : src(sv), pos(0) {}

    Tok next() {
        skipWS();
        ถ้า (pos >= src.size()) {
            คืน { END, "" };
        }

        var_na c = src[pos];

        // number
        ถ้า (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            ขณะ (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            คืน { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ถ้า (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            ขณะ (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            คืน { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ถ้า (c == '(') {
            pos++;
            คืน { LPAREN, "(" };
        }
        ถ้า (c == ')') {
            pos++;
            คืน { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        สำหรับ (auto op : two_ops) {
            ถ้า (src.substr(pos, 2) == op) {
                pos += 2;
                คืน { OP, std::string(op) };
            }
        }

        // single-char operators
        ถ้า (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            คืน { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        คืน { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        ขณะ (pos < src.size() && std::isspace((unsigned var_na) src[pos])) {
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

    จำนวนเต็ม parse() { คืน parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        ถ้า (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            คืน true;
        }
        คืน false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        ถ้า (tok.kind == k) {
            advance();
            คืน true;
        }
        คืน false;
    }

    จำนวนเต็ม parseLogicalOr() {
        จำนวนเต็ม v = parseLogicalAnd();
        ขณะ (acceptOp("||")) {
            จำนวนเต็ม rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        คืน v;
    }

    จำนวนเต็ม parseLogicalAnd() {
        จำนวนเต็ม v = parseEquality();
        ขณะ (acceptOp("&&")) {
            จำนวนเต็ม rhs = parseEquality();
            v       = (v && rhs);
        }
        คืน v;
    }

    จำนวนเต็ม parseEquality() {
        จำนวนเต็ม v = parseRelational();
        สำหรับ (;;) {
            ถ้า (acceptOp("==")) {
                จำนวนเต็ม rhs = parseRelational();
                v       = (v == rhs);
            } มิฉะนั้น ถ้า (acceptOp("!=")) {
                จำนวนเต็ม rhs = parseRelational();
                v       = (v != rhs);
            } มิฉะนั้น {
                bha_nga;
            }
        }
        คืน v;
    }

    จำนวนเต็ม parseRelational() {
        จำนวนเต็ม v = parseShift();
        สำหรับ (;;) {
            ถ้า (acceptOp("<")) {
                จำนวนเต็ม rhs = parseShift();
                v       = (v < rhs);
            } มิฉะนั้น ถ้า (acceptOp(">")) {
                จำนวนเต็ม rhs = parseShift();
                v       = (v > rhs);
            } มิฉะนั้น ถ้า (acceptOp("<=")) {
                จำนวนเต็ม rhs = parseShift();
                v       = (v <= rhs);
            } มิฉะนั้น ถ้า (acceptOp(">=")) {
                จำนวนเต็ม rhs = parseShift();
                v       = (v >= rhs);
            } มิฉะนั้น {
                bha_nga;
            }
        }
        คืน v;
    }

    จำนวนเต็ม parseShift() {
        จำนวนเต็ม v = parseAdd();
        สำหรับ (;;) {
            ถ้า (acceptOp("<<")) {
                จำนวนเต็ม rhs = parseAdd();
                v       = (v << rhs);
            } มิฉะนั้น ถ้า (acceptOp(">>")) {
                จำนวนเต็ม rhs = parseAdd();
                v       = (v >> rhs);
            } มิฉะนั้น {
                bha_nga;
            }
        }
        คืน v;
    }

    จำนวนเต็ม parseAdd() {
        จำนวนเต็ม v = parseMult();
        สำหรับ (;;) {
            ถ้า (acceptOp("+")) {
                จำนวนเต็ม rhs = parseMult();
                v       = (v + rhs);
            } มิฉะนั้น ถ้า (acceptOp("-")) {
                จำนวนเต็ม rhs = parseMult();
                v       = (v - rhs);
            } มิฉะนั้น {
                bha_nga;
            }
        }
        คืน v;
    }

    จำนวนเต็ม parseMult() {
        จำนวนเต็ม v = parseUnary();
        สำหรับ (;;) {
            ถ้า (acceptOp("*")) {
                จำนวนเต็ม rhs = parseUnary();
                v       = (v * rhs);
            } มิฉะนั้น ถ้า (acceptOp("/")) {
                จำนวนเต็ม rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } มิฉะนั้น ถ้า (acceptOp("%")) {
                จำนวนเต็ม rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } มิฉะนั้น {
                bha_nga;
            }
        }
        คืน v;
    }

    จำนวนเต็ม parseUnary() {
        ถ้า (acceptOp("!")) {
            คืน !parseUnary();
        }
        ถ้า (acceptOp("-")) {
            คืน -parseUnary();
        }
        ถ้า (acceptOp("+")) {
            คืน +parseUnary();
        }
        คืน parsePrimary();
    }

    จำนวนเต็ม parsePrimary() {
        // '(' expr ')'
        ถ้า (acceptKind(ExprLexer::LPAREN)) {
            จำนวนเต็ม v = parse();
            ถ้า (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            คืน v;
        }

        // number
        ถ้า (tok.kind == ExprLexer::NUMBER) {
            จำนวนเต็ม v = std::stoi(tok.text);
            advance();
            คืน v;
        }

        // defined(identifier)
        ถ้า (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            ถ้า (acceptKind(ExprLexer::LPAREN)) {
                ถ้า (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ถ้า (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                คืน macros.count(name) ? 1 : 0;
            } มิฉะนั้น {
                // defined NAME
                ถ้า (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                คืน macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ถ้า (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ถ้า (it == macros.end()) {
                คืน 0;
            }
            ถ้า (it->second.empty()) {
                คืน 1;
            }
            คืน evalMacroExpression(name, it->second);
        }

        // unexpected
        คืน 0;
    }

    จำนวนเต็ม evalMacroExpression(const std::string & name, const std::string & value) {
        ถ้า (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        จำนวนเต็ม        v = ep.parse();
        visiting.erase(name);
        คืน v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ถ้า (opts_.include_path.empty()) {
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
        คืน result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        คืน result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        คืน result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        คืน result;
    }

  private:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    ga_nanaa class DirectiveMode { All, IncludesOnly };

    sa_nracanaa Cond {
        bool parent_active;
        bool active;
        bool taken;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    void parseMacroDefinitions(const std::vector<std::string> & macro_defs) {
        สำหรับ (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ถ้า (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } มิฉะนั้น {
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

        สำหรับ (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        สำหรับ (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ถ้า (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } มิฉะนั้น {
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
        ถ้า (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        คืน ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        ถ้า (cond.empty()) {
            คืน true;
        }
        คืน cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        ถ้า (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        คืน out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        คืน processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        ขณะ (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            ถ้า (!t.empty() && t[0] == '#') {
                ขณะ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ถ้า (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            ถ้า (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                ถ้า (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } มิฉะนั้น {
                ถ้า (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } มิฉะนั้น ถ้า (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ถ้า (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        คืน out.str();
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

        ถ้า (cmd == "include") {
            ถ้า (mode == DirectiveMode::All && !condActive(cond)) {
                คืน true;
            }
            std::string file;
            iss >> file;
            ถ้า (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            คืน true;
        }

        ถ้า (mode == DirectiveMode::IncludesOnly) {
            คืน false;
        }

        ถ้า (cmd == "define") {
            ถ้า (!condActive(cond)) {
                คืน true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ถ้า (predefined_macros.count(name)) {
                คืน true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            คืน true;
        }

        ถ้า (cmd == "undef") {
            ถ้า (!condActive(cond)) {
                คืน true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ถ้า (predefined_macros.count(name)) {
                คืน true;
            }
            macros.erase(name);
            คืน true;
        }

        ถ้า (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            คืน true;
        }

        ถ้า (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            คืน true;
        }

        ถ้า (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            ถ้า (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            คืน true;
        }

        ถ้า (cmd == "elif") {
            std::string expr = trim_value(iss);

            ถ้า (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ถ้า (!c.parent_active) {
                c.active = false;
                คืน true;
            }

            ถ้า (c.taken) {
                c.active = false;
                คืน true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            ถ้า (v) {
                c.taken = true;
            }
            คืน true;
        }

        ถ้า (cmd == "else") {
            ถ้า (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ถ้า (!c.parent_active) {
                c.active = false;
                คืน true;
            }
            ถ้า (c.taken) {
                c.active = false;
            } มิฉะนั้น {
                c.active = true;
                c.taken  = true;
            }
            คืน true;
        }

        ถ้า (cmd == "endif") {
            ถ้า (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            คืน true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
