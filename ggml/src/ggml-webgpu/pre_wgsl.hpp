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
    ᬲᭂᬤᭂᬮ (a < s.size() && std::isspace((unsigned var_na) s[a])) {
        a++;
    }
    size_t b = s.size();
    ᬲᭂᬤᭂᬮ (b > a && std::isspace((unsigned var_na) s[b - 1])) {
        b--;
    }
    ᬯᬮᬶ s.substr(a, b - a);
}

static std::string trim_value(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ᬯᬮᬶ trim(ss.str());
}

static bool isIdentChar(var_na c) {
    ᬯᬮᬶ std::isalnum(static_cast<unsigned var_na>(c)) || c == '_';
}

static bool endsWithContinuation(const std::string & line) {
    size_t i = line.size();
    ᬲᭂᬤᭂᬮ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ᬯᬮᬶ i > 0 && line[i - 1] == '\\';
}

static void stripContinuation(std::string & line) {
    size_t i = line.size();
    ᬲᭂᬤᭂᬮ (i > 0 && std::isspace((unsigned var_na) line[i - 1])) {
        i--;
    }
    ᬬᬾᬦ᭄ (i > 0 && line[i - 1] == '\\') {
        line.erase(i - 1);
    }
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

static std::string expandMacroValue(const std::string &                                  name,
                                    const std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ᬬᬾᬦ᭄ (visiting.count(name)) {
        k_sipa std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    auto it = macros.find(name);
    ᬬᬾᬦ᭄ (it == macros.end()) {
        visiting.erase(name);
        ᬯᬮᬶ name;
    }

    const std::string & value = it->second;
    ᬬᬾᬦ᭄ (value.empty()) {
        visiting.erase(name);
        ᬯᬮᬶ "";
    }

    std::string expanded = expandMacrosRecursiveInternal(value, macros, visiting);
    visiting.erase(name);
    ᬯᬮᬶ expanded;
}

static std::string expandMacrosRecursiveInternal(const std::string &                                  line,
                                                 const std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string result;
    result.reserve(line.size());

    size_t i = 0;
    ᬲᭂᬤᭂᬮ (i < line.size()) {
        ᬬᬾᬦ᭄ (isIdentChar(line[i])) {
            size_t start = i;
            ᬲᭂᬤᭂᬮ (i < line.size() && isIdentChar(line[i])) {
                i++;
            }
            std::string token = line.substr(start, i - start);

            auto it = macros.find(token);
            ᬬᬾᬦ᭄ (it != macros.end()) {
                result += expandMacroValue(token, macros, visiting);
            } ᬮᬶᬬ {
                result += token;
            }
        } ᬮᬶᬬ {
            result += line[i];
            i++;
        }
    }

    ᬯᬮᬶ result;
}

static std::string expandMacrosRecursive(const std::string &                                  line,
                                         const std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ᬯᬮᬶ expandMacrosRecursiveInternal(line, macros, visiting);
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
        ᬬᬾᬦ᭄ (pos >= src.size()) {
            ᬯᬮᬶ { END, "" };
        }

        var_na c = src[pos];

        // number
        ᬬᬾᬦ᭄ (std::isdigit((unsigned var_na) c)) {
            size_t start = pos;
            ᬲᭂᬤᭂᬮ (pos < src.size() && std::isdigit((unsigned var_na) src[pos])) {
                pos++;
            }
            ᬯᬮᬶ { NUMBER, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ᬬᬾᬦ᭄ (std::isalpha((unsigned var_na) c) || c == '_') {
            size_t start = pos;
            ᬲᭂᬤᭂᬮ (pos < src.size() && (std::isalnum((unsigned var_na) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ᬯᬮᬶ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ᬬᬾᬦ᭄ (c == '(') {
            pos++;
            ᬯᬮᬶ { LPAREN, "(" };
        }
        ᬬᬾᬦ᭄ (c == ')') {
            pos++;
            ᬯᬮᬶ { RPAREN, ")" };
        }

        // multi-char operators
        static const var_na * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        for (auto op : two_ops) {
            ᬬᬾᬦ᭄ (src.substr(pos, 2) == op) {
                pos += 2;
                ᬯᬮᬶ { OP, std::string(op) };
            }
        }

        // single-char operators
        ᬬᬾᬦ᭄ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ᬯᬮᬶ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ᬯᬮᬶ { END, "" };
    }

  private:
    std::string_view src;
    size_t           pos;

    void skipWS() {
        ᬲᭂᬤᭂᬮ (pos < src.size() && std::isspace((unsigned var_na) src[pos])) {
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

    puur_naa_nka parse() { ᬯᬮᬶ parseLogicalOr(); }

  private:
    ExprLexer                                            lex;
    ExprLexer::Tok                                       tok;
    const std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    void advance() { tok = lex.next(); }

    bool acceptOp(const std::string & s) {
        ᬬᬾᬦ᭄ (tok.kind == ExprLexer::OP && tok.text == s) {
            advance();
            ᬯᬮᬶ true;
        }
        ᬯᬮᬶ false;
    }

    bool acceptKind(ExprLexer::Kind k) {
        ᬬᬾᬦ᭄ (tok.kind == k) {
            advance();
            ᬯᬮᬶ true;
        }
        ᬯᬮᬶ false;
    }

    puur_naa_nka parseLogicalOr() {
        puur_naa_nka v = parseLogicalAnd();
        ᬲᭂᬤᭂᬮ (acceptOp("||")) {
            puur_naa_nka rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ᬯᬮᬶ v;
    }

    puur_naa_nka parseLogicalAnd() {
        puur_naa_nka v = parseEquality();
        ᬲᭂᬤᭂᬮ (acceptOp("&&")) {
            puur_naa_nka rhs = parseEquality();
            v       = (v && rhs);
        }
        ᬯᬮᬶ v;
    }

    puur_naa_nka parseEquality() {
        puur_naa_nka v = parseRelational();
        for (;;) {
            ᬬᬾᬦ᭄ (acceptOp("==")) {
                puur_naa_nka rhs = parseRelational();
                v       = (v == rhs);
            } ᬮᬶᬬ ᬬᬾᬦ᭄ (acceptOp("!=")) {
                puur_naa_nka rhs = parseRelational();
                v       = (v != rhs);
            } ᬮᬶᬬ {
                bha_nga;
            }
        }
        ᬯᬮᬶ v;
    }

    puur_naa_nka parseRelational() {
        puur_naa_nka v = parseShift();
        for (;;) {
            ᬬᬾᬦ᭄ (acceptOp("<")) {
                puur_naa_nka rhs = parseShift();
                v       = (v < rhs);
            } ᬮᬶᬬ ᬬᬾᬦ᭄ (acceptOp(">")) {
                puur_naa_nka rhs = parseShift();
                v       = (v > rhs);
            } ᬮᬶᬬ ᬬᬾᬦ᭄ (acceptOp("<=")) {
                puur_naa_nka rhs = parseShift();
                v       = (v <= rhs);
            } ᬮᬶᬬ ᬬᬾᬦ᭄ (acceptOp(">=")) {
                puur_naa_nka rhs = parseShift();
                v       = (v >= rhs);
            } ᬮᬶᬬ {
                bha_nga;
            }
        }
        ᬯᬮᬶ v;
    }

    puur_naa_nka parseShift() {
        puur_naa_nka v = parseAdd();
        for (;;) {
            ᬬᬾᬦ᭄ (acceptOp("<<")) {
                puur_naa_nka rhs = parseAdd();
                v       = (v << rhs);
            } ᬮᬶᬬ ᬬᬾᬦ᭄ (acceptOp(">>")) {
                puur_naa_nka rhs = parseAdd();
                v       = (v >> rhs);
            } ᬮᬶᬬ {
                bha_nga;
            }
        }
        ᬯᬮᬶ v;
    }

    puur_naa_nka parseAdd() {
        puur_naa_nka v = parseMult();
        for (;;) {
            ᬬᬾᬦ᭄ (acceptOp("+")) {
                puur_naa_nka rhs = parseMult();
                v       = (v + rhs);
            } ᬮᬶᬬ ᬬᬾᬦ᭄ (acceptOp("-")) {
                puur_naa_nka rhs = parseMult();
                v       = (v - rhs);
            } ᬮᬶᬬ {
                bha_nga;
            }
        }
        ᬯᬮᬶ v;
    }

    puur_naa_nka parseMult() {
        puur_naa_nka v = parseUnary();
        for (;;) {
            ᬬᬾᬦ᭄ (acceptOp("*")) {
                puur_naa_nka rhs = parseUnary();
                v       = (v * rhs);
            } ᬮᬶᬬ ᬬᬾᬦ᭄ (acceptOp("/")) {
                puur_naa_nka rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ᬮᬶᬬ ᬬᬾᬦ᭄ (acceptOp("%")) {
                puur_naa_nka rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ᬮᬶᬬ {
                bha_nga;
            }
        }
        ᬯᬮᬶ v;
    }

    puur_naa_nka parseUnary() {
        ᬬᬾᬦ᭄ (acceptOp("!")) {
            ᬯᬮᬶ !parseUnary();
        }
        ᬬᬾᬦ᭄ (acceptOp("-")) {
            ᬯᬮᬶ -parseUnary();
        }
        ᬬᬾᬦ᭄ (acceptOp("+")) {
            ᬯᬮᬶ +parseUnary();
        }
        ᬯᬮᬶ parsePrimary();
    }

    puur_naa_nka parsePrimary() {
        // '(' expr ')'
        ᬬᬾᬦ᭄ (acceptKind(ExprLexer::LPAREN)) {
            puur_naa_nka v = parse();
            ᬬᬾᬦ᭄ (!acceptKind(ExprLexer::RPAREN)) {
                k_sipa std::runtime_error("missing ')'");
            }
            ᬯᬮᬶ v;
        }

        // number
        ᬬᬾᬦ᭄ (tok.kind == ExprLexer::NUMBER) {
            puur_naa_nka v = std::stoi(tok.text);
            advance();
            ᬯᬮᬶ v;
        }

        // defined(identifier)
        ᬬᬾᬦ᭄ (tok.kind == ExprLexer::IDENT && tok.text == "defined") {
            advance();
            ᬬᬾᬦ᭄ (acceptKind(ExprLexer::LPAREN)) {
                ᬬᬾᬦ᭄ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.text;
                advance();
                ᬬᬾᬦ᭄ (!acceptKind(ExprLexer::RPAREN)) {
                    k_sipa std::runtime_error("missing ) in defined()");
                }
                ᬯᬮᬶ macros.count(name) ? 1 : 0;
            } ᬮᬶᬬ {
                // defined NAME
                ᬬᬾᬦ᭄ (tok.kind != ExprLexer::IDENT) {
                    k_sipa std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.text;
                advance();
                ᬯᬮᬶ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ᬬᬾᬦ᭄ (tok.kind == ExprLexer::IDENT) {
            std::string name = tok.text;
            advance();
            auto it = macros.find(name);
            ᬬᬾᬦ᭄ (it == macros.end()) {
                ᬯᬮᬶ 0;
            }
            ᬬᬾᬦ᭄ (it->second.empty()) {
                ᬯᬮᬶ 1;
            }
            ᬯᬮᬶ evalMacroExpression(name, it->second);
        }

        // unexpected
        ᬯᬮᬶ 0;
    }

    puur_naa_nka evalMacroExpression(const std::string & name, const std::string & value) {
        ᬬᬾᬦ᭄ (visiting.count(name)) {
            k_sipa std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ExprParser ep(value, macros, visiting);
        puur_naa_nka        v = ep.parse();
        visiting.erase(name);
        ᬯᬮᬶ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
class Preprocessor {
  public:
    explicit Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ᬬᬾᬦ᭄ (opts_.include_path.empty()) {
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
        ᬯᬮᬶ result;
    }

    std::string preprocess(const std::string & contents, const std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        buildMacros(additional_macros, macros, predefined);

        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::All);
        ᬯᬮᬶ result;
    }

    std::string preprocess_includes_file(const std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processFile(filename, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ᬯᬮᬶ result;
    }

    std::string preprocess_includes(const std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_stack;
        std::string result = processString(contents, macros, predefined, include_stack, DirectiveMode::IncludesOnly);
        ᬯᬮᬶ result;
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
        for (const auto & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ᬬᬾᬦ᭄ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ᬮᬶᬬ {
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

        for (const auto & [name, value] : global_macros) {
            predefined.insert(name);
        }

        for (const auto & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ᬬᬾᬦ᭄ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ᬮᬶᬬ {
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
        ᬬᬾᬦ᭄ (!f.is_open()) {
            k_sipa std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ᬯᬮᬶ ss.str();
    }

    bool condActive(const std::vector<Cond> & cond) const {
        ᬬᬾᬦ᭄ (cond.empty()) {
            ᬯᬮᬶ true;
        }
        ᬯᬮᬶ cond.back().active;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(const std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            const std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_stack,
                            DirectiveMode                                  mode) {
        ᬬᬾᬦ᭄ (include_stack.count(name)) {
            k_sipa std::runtime_error("Recursive include: " + name);
        }

        include_stack.insert(name);
        std::string shader_code = loadFile(name);
        std::string out         = processString(shader_code, macros, predefined_macros, include_stack, mode);
        include_stack.erase(name);
        ᬯᬮᬶ out;
    }

    std::string processIncludeFile(const std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   const std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_stack,
                                   DirectiveMode                                  mode) {
        std::string full_path = opts_.include_path + "/" + fname;
        ᬯᬮᬶ processFile(full_path, macros, predefined_macros, include_stack, mode);
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

        ᬲᭂᬤᭂᬮ (std::getline(in, line)) {
            std::string logical = line;
            std::string t       = trim(logical);
            ᬬᬾᬦ᭄ (!t.empty() && t[0] == '#') {
                ᬲᭂᬤᭂᬮ (endsWithContinuation(logical)) {
                    stripContinuation(logical);
                    ᬬᬾᬦ᭄ (!std::getline(in, line)) {
                        bha_nga;
                    }
                    logical += "\n";
                    logical += line;
                }
                t = trim(logical);
            }

            ᬬᬾᬦ᭄ (!t.empty() && t[0] == '#') {
                bool handled = handleDirective(t, out, macros, predefined_macros, cond, include_stack, mode);
                ᬬᬾᬦ᭄ (mode == DirectiveMode::IncludesOnly && !handled) {
                    out << logical << "\n";
                }
            } ᬮᬶᬬ {
                ᬬᬾᬦ᭄ (mode == DirectiveMode::IncludesOnly) {
                    out << logical << "\n";
                } ᬮᬶᬬ ᬬᬾᬦ᭄ (condActive(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    out << expanded << "\n";
                }
            }
        }

        ᬬᬾᬦ᭄ (mode == DirectiveMode::All && !cond.empty()) {
            k_sipa std::runtime_error("Unclosed #if directive");
        }

        ᬯᬮᬶ out.str();
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

        ᬬᬾᬦ᭄ (cmd == "include") {
            ᬬᬾᬦ᭄ (mode == DirectiveMode::All && !condActive(cond)) {
                ᬯᬮᬶ true;
            }
            std::string file;
            iss >> file;
            ᬬᬾᬦ᭄ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            out << processIncludeFile(file, macros, predefined_macros, include_stack, mode);
            ᬯᬮᬶ true;
        }

        ᬬᬾᬦ᭄ (mode == DirectiveMode::IncludesOnly) {
            ᬯᬮᬶ false;
        }

        ᬬᬾᬦ᭄ (cmd == "define") {
            ᬬᬾᬦ᭄ (!condActive(cond)) {
                ᬯᬮᬶ true;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ᬬᬾᬦ᭄ (predefined_macros.count(name)) {
                ᬯᬮᬶ true;
            }
            std::string value = trim_value(iss);
            macros[name]      = value;
            ᬯᬮᬶ true;
        }

        ᬬᬾᬦ᭄ (cmd == "undef") {
            ᬬᬾᬦ᭄ (!condActive(cond)) {
                ᬯᬮᬶ true;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ᬬᬾᬦ᭄ (predefined_macros.count(name)) {
                ᬯᬮᬶ true;
            }
            macros.erase(name);
            ᬯᬮᬶ true;
        }

        ᬬᬾᬦ᭄ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ᬯᬮᬶ true;
        }

        ᬬᬾᬦ᭄ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            bool p = condActive(cond);
            bool v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ᬯᬮᬶ true;
        }

        ᬬᬾᬦ᭄ (cmd == "if") {
            std::string expr = trim_value(iss);
            bool        p    = condActive(cond);
            bool        v    = false;
            ᬬᬾᬦ᭄ (p) {
                std::unordered_set<std::string> visiting;
                ExprParser                      ep(expr, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ᬯᬮᬶ true;
        }

        ᬬᬾᬦ᭄ (cmd == "elif") {
            std::string expr = trim_value(iss);

            ᬬᬾᬦ᭄ (cond.empty()) {
                k_sipa std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ᬬᬾᬦ᭄ (!c.parent_active) {
                c.active = false;
                ᬯᬮᬶ true;
            }

            ᬬᬾᬦ᭄ (c.taken) {
                c.active = false;
                ᬯᬮᬶ true;
            }

            std::unordered_set<std::string> visiting;
            ExprParser                      ep(expr, macros, visiting);
            bool                            v = ep.parse() != 0;
            c.active                          = v;
            ᬬᬾᬦ᭄ (v) {
                c.taken = true;
            }
            ᬯᬮᬶ true;
        }

        ᬬᬾᬦ᭄ (cmd == "else") {
            ᬬᬾᬦ᭄ (cond.empty()) {
                k_sipa std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ᬬᬾᬦ᭄ (!c.parent_active) {
                c.active = false;
                ᬯᬮᬶ true;
            }
            ᬬᬾᬦ᭄ (c.taken) {
                c.active = false;
            } ᬮᬶᬬ {
                c.active = true;
                c.taken  = true;
            }
            ᬯᬮᬶ true;
        }

        ᬬᬾᬦ᭄ (cmd == "endif") {
            ᬬᬾᬦ᭄ (cond.empty()) {
                k_sipa std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ᬯᬮᬶ true;
        }

        // Unknown directive
        k_sipa std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#endif  // PRE_WGSL_HPP
