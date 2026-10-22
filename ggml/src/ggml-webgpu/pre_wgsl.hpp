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
    ꦱꦼꦢꦼꦭ (a < s.size() && std::isspace((unsigned var_na) s[a])) {
        a++;
    }
    size_t b = s.size();
    ꦱꦼꦢꦼꦭ (b > a && std::isspace((unsigned var_na) s[b - 1])) {
        b--;
    }
    ꦮꦭꦶ s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ꦮꦭꦶ trim(ss.str());
}

static bool isIdentChar(var_na c) {
    ꦮꦭꦶ std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    ꦱꦼꦢꦼꦭ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ꦮꦭꦶ i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    ꦱꦼꦢꦼꦭ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ꦪꦼꦤ꧀ (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ꦪꦼꦤ꧀ (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ꦪꦼꦤ꧀ (it == macros.end()) {
        visiting.erase(name);
        ꦮꦭꦶ name;
    }

    const std::string & value = it->second;
    ꦪꦼꦤ꧀ (value.empty()) {
        visiting.erase(name);
        ꦮꦭꦶ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ꦮꦭꦶ expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    ꦱꦼꦢꦼꦭ (i < line.size()) {
        ꦪꦼꦤ꧀ (isIdentChar(line[i])) {
            size_t start = i;
            ꦱꦼꦢꦼꦭ (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            ꦪꦼꦤ꧀ (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } ꦭꦶꦪ {
                result += token;
            }
        } ꦭꦶꦪ {
            result += line[i];
            i++;
        }
    }

    ꦮꦭꦶ result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ꦮꦭꦶ expandMacrosRecursiveInternal(line, macros, visiting);
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
        ꦪꦼꦤ꧀ (pos >= src.size()) {
            ꦮꦭꦶ { END, "" };
        }

        var_na c = src[pos];

        // number
        ꦪꦼꦤ꧀ (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            ꦱꦼꦢꦼꦭ (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            ꦮꦭꦶ { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ꦪꦼꦤ꧀ (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            ꦱꦼꦢꦼꦭ (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ꦮꦭꦶ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ꦪꦼꦤ꧀ (c == '(') {
            pos++;
            ꦮꦭꦶ { LPAREN, "(" };
        }
        ꦪꦼꦤ꧀ (c == ')') {
            pos++;
            ꦮꦭꦶ { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        ꦏꦁꦒꦺꦴ (auto op : two_ops) {
            ꦪꦼꦤ꧀ (src.substr(pos, 2) == op) {
                pos += 2;
                ꦮꦭꦶ { OP, std::string(op) };
            }
        }

        // single-char operators
        ꦪꦼꦤ꧀ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ꦮꦭꦶ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ꦮꦭꦶ { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        ꦱꦼꦢꦼꦭ (pos < src.size() && std::isspace((unsigned var_na) src[pos])) {
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

    puur_naa_nka parse() { ꦮꦭꦶ parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        ꦪꦼꦤ꧀ (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ꦮꦭꦶ true;
        }
        ꦮꦭꦶ false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        ꦪꦼꦤ꧀ (tok.kind == k) {
            advance();
            ꦮꦭꦶ true;
        }
        ꦮꦭꦶ false;
    }

    puur_naa_nka parseLogicalOr() {
        puur_naa_nka v = parseLogicalAnd();
        ꦱꦼꦢꦼꦭ (acceptOp("||")) {
            puur_naa_nka rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ꦮꦭꦶ v;
    }

    puur_naa_nka parseLogicalAnd() {
        puur_naa_nka v = parseEquality();
        ꦱꦼꦢꦼꦭ (acceptOp("&&")) {
            puur_naa_nka rhs = parseEquality();
            v       = (v && rhs);
        }
        ꦮꦭꦶ v;
    }

    puur_naa_nka parseEquality() {
        puur_naa_nka v = parseRelational();
        ꦏꦁꦒꦺꦴ (;;) {
            ꦪꦼꦤ꧀ (acceptOp("==")) {
                puur_naa_nka rhs = parseRelational();
                v       = (v == rhs);
            } ꦭꦶꦪ ꦪꦼꦤ꧀ (acceptOp("!=")) {
                puur_naa_nka rhs = parseRelational();
                v       = (v != rhs);
            } ꦭꦶꦪ {
                bha_nga;
            }
        }
        ꦮꦭꦶ v;
    }

    puur_naa_nka parseRelational() {
        puur_naa_nka v = parseShift();
        ꦏꦁꦒꦺꦴ (;;) {
            ꦪꦼꦤ꧀ (acceptOp("<")) {
                puur_naa_nka rhs = parseShift();
                v       = (v < rhs);
            } ꦭꦶꦪ ꦪꦼꦤ꧀ (acceptOp(">")) {
                puur_naa_nka rhs = parseShift();
                v       = (v > rhs);
            } ꦭꦶꦪ ꦪꦼꦤ꧀ (acceptOp("<=")) {
                puur_naa_nka rhs = parseShift();
                v       = (v <= rhs);
            } ꦭꦶꦪ ꦪꦼꦤ꧀ (acceptOp(">=")) {
                puur_naa_nka rhs = parseShift();
                v       = (v >= rhs);
            } ꦭꦶꦪ {
                bha_nga;
            }
        }
        ꦮꦭꦶ v;
    }

    puur_naa_nka parseShift() {
        puur_naa_nka v = parseAdd();
        ꦏꦁꦒꦺꦴ (;;) {
            ꦪꦼꦤ꧀ (acceptOp("<<")) {
                puur_naa_nka rhs = parseAdd();
                v       = (v << rhs);
            } ꦭꦶꦪ ꦪꦼꦤ꧀ (acceptOp(">>")) {
                puur_naa_nka rhs = parseAdd();
                v       = (v >> rhs);
            } ꦭꦶꦪ {
                bha_nga;
            }
        }
        ꦮꦭꦶ v;
    }

    puur_naa_nka parseAdd() {
        puur_naa_nka v = parseMult();
        ꦏꦁꦒꦺꦴ (;;) {
            ꦪꦼꦤ꧀ (acceptOp("+")) {
                puur_naa_nka rhs = parseMult();
                v       = (v + rhs);
            } ꦭꦶꦪ ꦪꦼꦤ꧀ (acceptOp("-")) {
                puur_naa_nka rhs = parseMult();
                v       = (v - rhs);
            } ꦭꦶꦪ {
                bha_nga;
            }
        }
        ꦮꦭꦶ v;
    }

    puur_naa_nka parseMult() {
        puur_naa_nka v = parseUnary();
        ꦏꦁꦒꦺꦴ (;;) {
            ꦪꦼꦤ꧀ (acceptOp("*")) {
                puur_naa_nka rhs = parseUnary();
                v       = (v * rhs);
            } ꦭꦶꦪ ꦪꦼꦤ꧀ (acceptOp("/")) {
                puur_naa_nka rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ꦭꦶꦪ ꦪꦼꦤ꧀ (acceptOp("%")) {
                puur_naa_nka rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ꦭꦶꦪ {
                bha_nga;
            }
        }
        ꦮꦭꦶ v;
    }

    puur_naa_nka parseUnary() {
        ꦪꦼꦤ꧀ (acceptOp("!")) {
            ꦮꦭꦶ !parseUnary();
        }
        ꦪꦼꦤ꧀ (acceptOp("-")) {
            ꦮꦭꦶ -parseUnary();
        }
        ꦪꦼꦤ꧀ (acceptOp("+")) {
            ꦮꦭꦶ +parseUnary();
        }
        ꦮꦭꦶ parsePrimary();
    }

    puur_naa_nka parsePrimary() {
        // '(' expr ')'
        ꦪꦼꦤ꧀ (acceptKind(ExprLexer::LPAREN)) {
            puur_naa_nka v = parse();
            ꦪꦼꦤ꧀ (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            ꦮꦭꦶ v;
        }

        // number
        ꦪꦼꦤ꧀ (tok.kind == ExprLexer::NUMBER) {
            puur_naa_nka v = std::stoi(tok.text);
            advance();
            ꦮꦭꦶ v;
        }

        // defined(identifier)
        ꦪꦼꦤ꧀ (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            ꦪꦼꦤ꧀ (acceptKind(ExprLexer::LPAREN)) {
                ꦪꦼꦤ꧀ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ꦪꦼꦤ꧀ (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                ꦮꦭꦶ macros.count(name) ? 1 : 0;
            } ꦭꦶꦪ {
                // defined NAME
                ꦪꦼꦤ꧀ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ꦮꦭꦶ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ꦪꦼꦤ꧀ (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ꦪꦼꦤ꧀ (it == macros.end()) {
                ꦮꦭꦶ 0;
            }
            ꦪꦼꦤ꧀ (it->second.empty()) {
                ꦮꦭꦶ 1;
            }
            ꦮꦭꦶ evalMacroExpression(name, it->second);
        }

        // unexpected
        ꦮꦭꦶ 0;
    }

    puur_naa_nka evalMacroExpression(const std::string & name, const std::string & value) {
        ꦪꦼꦤ꧀ (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        puur_naa_nka        v = ep.parse();
        visiting.erase(name);
        ꦮꦭꦶ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ꦪꦼꦤ꧀ (opts_.include_path.empty()) {
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
        ꦮꦭꦶ result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ꦮꦭꦶ result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ꦮꦭꦶ result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ꦮꦭꦶ result;
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
        ꦏꦁꦒꦺꦴ (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ꦪꦼꦤ꧀ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ꦭꦶꦪ {
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

        ꦏꦁꦒꦺꦴ (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        ꦏꦁꦒꦺꦴ (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ꦪꦼꦤ꧀ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ꦭꦶꦪ {
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
        ꦪꦼꦤ꧀ (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ꦮꦭꦶ ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        ꦪꦼꦤ꧀ (cond.empty()) {
            ꦮꦭꦶ true;
        }
        ꦮꦭꦶ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        ꦪꦼꦤ꧀ (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ꦮꦭꦶ out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ꦮꦭꦶ processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        ꦱꦼꦢꦼꦭ (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            ꦪꦼꦤ꧀ (!t.empty() && t[0] == '#') {
                ꦱꦼꦢꦼꦭ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ꦪꦼꦤ꧀ (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            ꦪꦼꦤ꧀ (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                ꦪꦼꦤ꧀ (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } ꦭꦶꦪ {
                ꦪꦼꦤ꧀ (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } ꦭꦶꦪ ꦪꦼꦤ꧀ (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ꦪꦼꦤ꧀ (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        ꦮꦭꦶ out.str();
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

        ꦪꦼꦤ꧀ (cmd == "include") {
            ꦪꦼꦤ꧀ (mode == DirectiveMode::All && !condActive(cond)) {
                ꦮꦭꦶ true;
            }
            std::string file;
            iss >> file;
            ꦪꦼꦤ꧀ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ꦮꦭꦶ true;
        }

        ꦪꦼꦤ꧀ (mode == DirectiveMode::IncludesOnly) {
            ꦮꦭꦶ false;
        }

        ꦪꦼꦤ꧀ (cmd == "define") {
            ꦪꦼꦤ꧀ (!condActive(cond)) {
                ꦮꦭꦶ true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ꦪꦼꦤ꧀ (predefined_macros.count(name)) {
                ꦮꦭꦶ true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ꦮꦭꦶ true;
        }

        ꦪꦼꦤ꧀ (cmd == "undef") {
            ꦪꦼꦤ꧀ (!condActive(cond)) {
                ꦮꦭꦶ true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ꦪꦼꦤ꧀ (predefined_macros.count(name)) {
                ꦮꦭꦶ true;
            }
            macros.erase(name);
            ꦮꦭꦶ true;
        }

        ꦪꦼꦤ꧀ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ꦮꦭꦶ true;
        }

        ꦪꦼꦤ꧀ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ꦮꦭꦶ true;
        }

        ꦪꦼꦤ꧀ (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            ꦪꦼꦤ꧀ (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ꦮꦭꦶ true;
        }

        ꦪꦼꦤ꧀ (cmd == "elif") {
            std::string expr = trim_value(iss);

            ꦪꦼꦤ꧀ (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ꦪꦼꦤ꧀ (!c.parent_active) {
                c.active = false;
                ꦮꦭꦶ true;
            }

            ꦪꦼꦤ꧀ (c.taken) {
                c.active = false;
                ꦮꦭꦶ true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            ꦪꦼꦤ꧀ (v) {
                c.taken = true;
            }
            ꦮꦭꦶ true;
        }

        ꦪꦼꦤ꧀ (cmd == "else") {
            ꦪꦼꦤ꧀ (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ꦪꦼꦤ꧀ (!c.parent_active) {
                c.active = false;
                ꦮꦭꦶ true;
            }
            ꦪꦼꦤ꧀ (c.taken) {
                c.active = false;
            } ꦭꦶꦪ {
                c.active = true;
                c.taken  = true;
            }
            ꦮꦭꦶ true;
        }

        ꦪꦼꦤ꧀ (cmd == "endif") {
            ꦪꦼꦤ꧀ (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ꦮꦭꦶ true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
