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
    අතරතුර (a < s.size() && std::isspace((unsigned var_na) s[a])) {
        a++;
    }
    size_t b = s.size();
    අතරතුර (b > a && std::isspace((unsigned var_na) s[b - 1])) {
        b--;
    }
    ආපසු s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ආපසු trim(ss.str());
}

static bool isIdentChar(var_na c) {
    ආපසු std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    අතරතුර (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ආපසු i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    අතරතුර (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    නම් (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    නම් (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    නම් (it == macros.end()) {
        visiting.erase(name);
        ආපසු name;
    }

    const std::string & value = it->second;
    නම් (value.empty()) {
        visiting.erase(name);
        ආපසු "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ආපසු expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    අතරතුර (i < line.size()) {
        නම් (isIdentChar(line[i])) {
            size_t start = i;
            අතරතුර (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            නම් (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } නැතහොත් {
                result += token;
            }
        } නැතහොත් {
            result += line[i];
            i++;
        }
    }

    ආපසු result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ආපසු expandMacrosRecursiveInternal(line, macros, visiting);
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
        නම් (pos >= src.size()) {
            ආපසු { END, "" };
        }

        var_na c = src[pos];

        // number
        නම් (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            අතරතුර (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            ආපසු { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        නම් (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            අතරතුර (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ආපසු { IDENT, std::string(src.substr(start, pos - start)) };
        }

        නම් (c == '(') {
            pos++;
            ආපසු { LPAREN, "(" };
        }
        නම් (c == ')') {
            pos++;
            ආපසු { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        සඳහා (auto op : two_ops) {
            නම් (src.substr(pos, 2) == op) {
                pos += 2;
                ආපසු { OP, std::string(op) };
            }
        }

        // single-char operators
        නම් (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ආපසු { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ආපසු { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        අතරතුර (pos < src.size() && std::isspace((unsigned var_na) src[pos])) {
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

    පූර්ණ_සංඛ්‍යා parse() { ආපසු parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        නම් (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ආපසු true;
        }
        ආපසු false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        නම් (tok.kind == k) {
            advance();
            ආපසු true;
        }
        ආපසු false;
    }

    පූර්ණ_සංඛ්‍යා parseLogicalOr() {
        පූර්ණ_සංඛ්‍යා v = parseLogicalAnd();
        අතරතුර (acceptOp("||")) {
            පූර්ණ_සංඛ්‍යා rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ආපසු v;
    }

    පූර්ණ_සංඛ්‍යා parseLogicalAnd() {
        පූර්ණ_සංඛ්‍යා v = parseEquality();
        අතරතුර (acceptOp("&&")) {
            පූර්ණ_සංඛ්‍යා rhs = parseEquality();
            v       = (v && rhs);
        }
        ආපසු v;
    }

    පූර්ණ_සංඛ්‍යා parseEquality() {
        පූර්ණ_සංඛ්‍යා v = parseRelational();
        සඳහා (;;) {
            නම් (acceptOp("==")) {
                පූර්ණ_සංඛ්‍යා rhs = parseRelational();
                v       = (v == rhs);
            } නැතහොත් නම් (acceptOp("!=")) {
                පූර්ණ_සංඛ්‍යා rhs = parseRelational();
                v       = (v != rhs);
            } නැතහොත් {
                bha_nga;
            }
        }
        ආපසු v;
    }

    පූර්ණ_සංඛ්‍යා parseRelational() {
        පූර්ණ_සංඛ්‍යා v = parseShift();
        සඳහා (;;) {
            නම් (acceptOp("<")) {
                පූර්ණ_සංඛ්‍යා rhs = parseShift();
                v       = (v < rhs);
            } නැතහොත් නම් (acceptOp(">")) {
                පූර්ණ_සංඛ්‍යා rhs = parseShift();
                v       = (v > rhs);
            } නැතහොත් නම් (acceptOp("<=")) {
                පූර්ණ_සංඛ්‍යා rhs = parseShift();
                v       = (v <= rhs);
            } නැතහොත් නම් (acceptOp(">=")) {
                පූර්ණ_සංඛ්‍යා rhs = parseShift();
                v       = (v >= rhs);
            } නැතහොත් {
                bha_nga;
            }
        }
        ආපසු v;
    }

    පූර්ණ_සංඛ්‍යා parseShift() {
        පූර්ණ_සංඛ්‍යා v = parseAdd();
        සඳහා (;;) {
            නම් (acceptOp("<<")) {
                පූර්ණ_සංඛ්‍යා rhs = parseAdd();
                v       = (v << rhs);
            } නැතහොත් නම් (acceptOp(">>")) {
                පූර්ණ_සංඛ්‍යා rhs = parseAdd();
                v       = (v >> rhs);
            } නැතහොත් {
                bha_nga;
            }
        }
        ආපසු v;
    }

    පූර්ණ_සංඛ්‍යා parseAdd() {
        පූර්ණ_සංඛ්‍යා v = parseMult();
        සඳහා (;;) {
            නම් (acceptOp("+")) {
                පූර්ණ_සංඛ්‍යා rhs = parseMult();
                v       = (v + rhs);
            } නැතහොත් නම් (acceptOp("-")) {
                පූර්ණ_සංඛ්‍යා rhs = parseMult();
                v       = (v - rhs);
            } නැතහොත් {
                bha_nga;
            }
        }
        ආපසු v;
    }

    පූර්ණ_සංඛ්‍යා parseMult() {
        පූර්ණ_සංඛ්‍යා v = parseUnary();
        සඳහා (;;) {
            නම් (acceptOp("*")) {
                පූර්ණ_සංඛ්‍යා rhs = parseUnary();
                v       = (v * rhs);
            } නැතහොත් නම් (acceptOp("/")) {
                පූර්ණ_සංඛ්‍යා rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } නැතහොත් නම් (acceptOp("%")) {
                පූර්ණ_සංඛ්‍යා rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } නැතහොත් {
                bha_nga;
            }
        }
        ආපසු v;
    }

    පූර්ණ_සංඛ්‍යා parseUnary() {
        නම් (acceptOp("!")) {
            ආපසු !parseUnary();
        }
        නම් (acceptOp("-")) {
            ආපසු -parseUnary();
        }
        නම් (acceptOp("+")) {
            ආපසු +parseUnary();
        }
        ආපසු parsePrimary();
    }

    පූර්ණ_සංඛ්‍යා parsePrimary() {
        // '(' expr ')'
        නම් (acceptKind(ExprLexer::LPAREN)) {
            පූර්ණ_සංඛ්‍යා v = parse();
            නම් (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            ආපසු v;
        }

        // number
        නම් (tok.kind == ExprLexer::NUMBER) {
            පූර්ණ_සංඛ්‍යා v = std::stoi(tok.text);
            advance();
            ආපසු v;
        }

        // defined(identifier)
        නම් (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            නම් (acceptKind(ExprLexer::LPAREN)) {
                නම් (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                නම් (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                ආපසු macros.count(name) ? 1 : 0;
            } නැතහොත් {
                // defined NAME
                නම් (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ආපසු macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        නම් (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            නම් (it == macros.end()) {
                ආපසු 0;
            }
            නම් (it->second.empty()) {
                ආපසු 1;
            }
            ආපසු evalMacroExpression(name, it->second);
        }

        // unexpected
        ආපසු 0;
    }

    පූර්ණ_සංඛ්‍යා evalMacroExpression(const std::string & name, const std::string & value) {
        නම් (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        පූර්ණ_සංඛ්‍යා        v = ep.parse();
        visiting.erase(name);
        ආපසු v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        නම් (opts_.include_path.empty()) {
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
        ආපසු result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ආපසු result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ආපසු result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ආපසු result;
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
        සඳහා (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            නම් (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } නැතහොත් {
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

        සඳහා (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        සඳහා (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            නම් (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } නැතහොත් {
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
        නම් (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ආපසු ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        නම් (cond.empty()) {
            ආපසු true;
        }
        ආපසු cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        නම් (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ආපසු out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ආපසු processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        අතරතුර (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            නම් (!t.empty() && t[0] == '#') {
                අතරතුර (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    නම් (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            නම් (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                නම් (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } නැතහොත් {
                නම් (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } නැතහොත් නම් (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        නම් (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        ආපසු out.str();
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

        නම් (cmd == "include") {
            නම් (mode == DirectiveMode::All && !condActive(cond)) {
                ආපසු true;
            }
            std::string file;
            iss >> file;
            නම් (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ආපසු true;
        }

        නම් (mode == DirectiveMode::IncludesOnly) {
            ආපසු false;
        }

        නම් (cmd == "define") {
            නම් (!condActive(cond)) {
                ආපසු true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            නම් (predefined_macros.count(name)) {
                ආපසු true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ආපසු true;
        }

        නම් (cmd == "undef") {
            නම් (!condActive(cond)) {
                ආපසු true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            නම් (predefined_macros.count(name)) {
                ආපසු true;
            }
            macros.erase(name);
            ආපසු true;
        }

        නම් (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ආපසු true;
        }

        නම් (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ආපසු true;
        }

        නම් (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            නම් (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ආපසු true;
        }

        නම් (cmd == "elif") {
            std::string expr = trim_value(iss);

            නම් (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            නම් (!c.parent_active) {
                c.active = false;
                ආපසු true;
            }

            නම් (c.taken) {
                c.active = false;
                ආපසු true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            නම් (v) {
                c.taken = true;
            }
            ආපසු true;
        }

        නම් (cmd == "else") {
            නම් (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            නම් (!c.parent_active) {
                c.active = false;
                ආපසු true;
            }
            නම් (c.taken) {
                c.active = false;
            } නැතහොත් {
                c.active = true;
                c.taken  = true;
            }
            ආපසු true;
        }

        නම් (cmd == "endif") {
            නම් (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ආපසු true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
