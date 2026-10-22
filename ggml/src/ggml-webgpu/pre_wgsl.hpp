#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ PRE_WGSL_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು PRE_WGSL_HPP

#ಅನ್ತರ್ಭಾವ <cctype>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <sstream>
#ಅನ್ತರ್ಭಾವ <stdexcept>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <string_view>
#ಅನ್ತರ್ಭಾವ <unordered_map>
#ಅನ್ತರ್ಭಾವ <unordered_set>
#ಅನ್ತರ್ಭಾವ <vector>

ನಾಮಾಕಾಶ pre_wgsl {

//==============================================================
// Options
//==============================================================
ರಚನೆ Options {
    std::string              include_ಮಾರ್ಗ = ".";
    std::vector<std::string> macros;
};

//==============================================================
// Utility: trim
//==============================================================
ಸ್ಥಿರ std::string trim(ಸ್ಥಿರಾಂಕ std::string & s) {
    size_t a = 0;
    ಆಗ (a < s.size() && std::ಖಾಲಿಯೇ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ) s[a])) {
        a++;
    }
    size_t b = s.size();
    ಆಗ (b > a && std::ಖಾಲಿಯೇ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ) s[b - 1])) {
        b--;
    }
    ಹಿಂತಿರುಗಿ s.substr(a, b - a);
}

ಸ್ಥಿರ std::string trim_ಮೌಲ್ಯ(std::istream & is) {
    std::ostringstream ss;
    ss << is.rdbuf();
    ಹಿಂತಿರುಗಿ trim(ss.str());
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ isIdentChar(ಅಕ್ಷರ c) {
    ಹಿಂತಿರುಗಿ std::ಅಕ್ಷರವೇ_ಅಂಕಿಯೇ(static_cast<ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ>(c)) || c == '_';
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ತುದಿಗಳು_With_Continuation(ಸ್ಥಿರಾಂಕ std::string & ಸಾಲು) {
    size_t i = ಸಾಲು.size();
    ಆಗ (i > 0 && std::ಖಾಲಿಯೇ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ) ಸಾಲು[i - 1])) {
        i--;
    }
    ಹಿಂತಿರುಗಿ i > 0 && ಸಾಲು[i - 1] == '\\';
}

ಸ್ಥಿರ ಶೂನ್ಯ stripContinuation(std::string & ಸಾಲು) {
    size_t i = ಸಾಲು.size();
    ಆಗ (i > 0 && std::ಖಾಲಿಯೇ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ) ಸಾಲು[i - 1])) {
        i--;
    }
    ಆದರೆ (i > 0 && ಸಾಲು[i - 1] == '\\') {
        ಸಾಲು.erase(i - 1);
    }
}

ಸ್ಥಿರ std::string expand_Macros_Recursive_ಆಂತರಿಕ(ಸ್ಥಿರಾಂಕ std::string &                                  ಸಾಲು,
                                                 ಸ್ಥಿರಾಂಕ std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting);

ಸ್ಥಿರ std::string expand_Macro_ಮೌಲ್ಯ(ಸ್ಥಿರಾಂಕ std::string &                                  name,
                                    ಸ್ಥಿರಾಂಕ std::unordered_map<std::string, std::string> & macros,
                                    std::unordered_set<std::string> &                    visiting) {
    ಆದರೆ (visiting.count(name)) {
        ಎಸೆ std::runtime_error("Recursive macro: " + name);
    }
    visiting.insert(name);

    ಸ್ವಯಂಚಾಲಿತ it = macros.find(name);
    ಆದರೆ (it == macros.end()) {
        visiting.erase(name);
        ಹಿಂತಿರುಗಿ name;
    }

    ಸ್ಥಿರಾಂಕ std::string & value = it->second;
    ಆದರೆ (value.ಖಾಲಿಗಳು()) {
        visiting.erase(name);
        ಹಿಂತಿರುಗಿ "";
    }

    std::string expanded = expand_Macros_Recursive_ಆಂತರಿಕ(value, macros, visiting);
    visiting.erase(name);
    ಹಿಂತಿರುಗಿ expanded;
}

ಸ್ಥಿರ std::string expand_Macros_Recursive_ಆಂತರಿಕ(ಸ್ಥಿರಾಂಕ std::string &                                  ಸಾಲು,
                                                 ಸ್ಥಿರಾಂಕ std::unordered_map<std::string, std::string> & macros,
                                                 std::unordered_set<std::string> &                    visiting) {
    std::string ಫಲಿತಾಂಶ;
    ಫಲಿತಾಂಶ.reserve(ಸಾಲು.size());

    size_t i = 0;
    ಆಗ (i < ಸಾಲು.size()) {
        ಆದರೆ (isIdentChar(ಸಾಲು[i])) {
            size_t start = i;
            ಆಗ (i < ಸಾಲು.size() && isIdentChar(ಸಾಲು[i])) {
                i++;
            }
            std::string token = ಸಾಲು.substr(start, i - start);

            ಸ್ವಯಂಚಾಲಿತ it = macros.find(token);
            ಆದರೆ (it != macros.end()) {
                ಫಲಿತಾಂಶ += expand_Macro_ಮೌಲ್ಯ(token, macros, visiting);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಫಲಿತಾಂಶ += token;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಫಲಿತಾಂಶ += ಸಾಲು[i];
            i++;
        }
    }

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸ್ಥಿರ std::string expandMacrosRecursive(ಸ್ಥಿರಾಂಕ std::string &                                  ಸಾಲು,
                                         ಸ್ಥಿರಾಂಕ std::unordered_map<std::string, std::string> & macros) {
    std::unordered_set<std::string> visiting;
    ಹಿಂತಿರುಗಿ expand_Macros_Recursive_ಆಂತರಿಕ(ಸಾಲು, macros, visiting);
}

//==============================================================
// Tokenizer for expressions in #if/#elif
//==============================================================
ವರ್ಗ ಅಭಿವ್ಯಕ್ತಿ_Lexer {
  ಸಾರ್ವಜನಿಕ:
    ಎಣಿಕೆ ಬಗೆ_2 { ಕೊನೆ_3, IDENT, ಸಂಖ್ಯೆ_2, OP, LPAREN, RPAREN };

    ರಚನೆ Tok {
        ಬಗೆ_2        ಬಗೆ;
        std::string ಪಠ್ಯ_2;
    };

    ಸ್ಪಷ್ಟ ಅಭಿವ್ಯಕ್ತಿ_Lexer(std::string_view sv) : src(sv), pos(0) {}

    Tok ಮುಂದಿನ() {
        skipWS();
        ಆದರೆ (pos >= src.size()) {
            ಹಿಂತಿರುಗಿ { ಕೊನೆ_3, "" };
        }

        ಅಕ್ಷರ c = src[pos];

        //ಸಂಖ್ಯೆ
        ಆದರೆ (std::ಅಂಕಿಯೇ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ) c)) {
            size_t start = pos;
            ಆಗ (pos < src.size() && std::ಅಂಕಿಯೇ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ) src[pos])) {
                pos++;
            }
            ಹಿಂತಿರುಗಿ { ಸಂಖ್ಯೆ_2, std::string(src.substr(start, pos - start)) };
        }

        // identifier
        ಆದರೆ (std::ಅಕ್ಷರವೇ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ) c) || c == '_') {
            size_t start = pos;
            ಆಗ (pos < src.size() && (std::ಅಕ್ಷರವೇ_ಅಂಕಿಯೇ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ) src[pos]) || src[pos] == '_')) {
                pos++;
            }
            ಹಿಂತಿರುಗಿ { IDENT, std::string(src.substr(start, pos - start)) };
        }

        ಆದರೆ (c == '(') {
            pos++;
            ಹಿಂತಿರುಗಿ { LPAREN, "(" };
        }
        ಆದರೆ (c == ')') {
            pos++;
            ಹಿಂತಿರುಗಿ { RPAREN, ")" };
        }

        // multi-char operators
        ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * two_ops[] = { "==", "!=", "<=", ">=", "&&", "||", "<<", ">>" };
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ op : two_ops) {
            ಆದರೆ (src.substr(pos, 2) == op) {
                pos += 2;
                ಹಿಂತಿರುಗಿ { OP, std::string(op) };
            }
        }

        // single-char operators
        ಆದರೆ (std::string("+-*/%<>!").find(c) != std::string::npos) {
            pos++;
            ಹಿಂತಿರುಗಿ { OP, std::string(1, c) };
        }

        // unexpected
        pos++;
        ಹಿಂತಿರುಗಿ { ಕೊನೆ_3, "" };
    }

  ಖಾಸಗಿ:
    std::string_view src;
    size_t           pos;

    ಶೂನ್ಯ skipWS() {
        ಆಗ (pos < src.size() && std::ಖಾಲಿಯೇ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ) src[pos])) {
            pos++;
        }
    }
};

//==============================================================
// Expression Parser (recursive descent)
//==============================================================
ವರ್ಗ ಅಭಿವ್ಯಕ್ತಿ_Parser {
  ಸಾರ್ವಜನಿಕ:
    ಅಭಿವ್ಯಕ್ತಿ_Parser(std::string_view                                     ಅಭಿವ್ಯಕ್ತಿ,
               ಸ್ಥಿರಾಂಕ std::unordered_map<std::string, std::string> & macros,
               std::unordered_set<std::string> &                    visiting) :
        lex(ಅಭಿವ್ಯಕ್ತಿ),
        macros(macros),
        visiting(visiting) {
        advance();
    }

    ಪೂರ್ಣಾಂಕ parse() { ಹಿಂತಿರುಗಿ parseLogicalOr(); }

  ಖಾಸಗಿ:
    ಅಭಿವ್ಯಕ್ತಿ_Lexer                                            lex;
    ಅಭಿವ್ಯಕ್ತಿ_Lexer::Tok                                       tok;
    ಸ್ಥಿರಾಂಕ std::unordered_map<std::string, std::string> & macros;
    std::unordered_set<std::string> &                    visiting;

    ಶೂನ್ಯ advance() { tok = lex.ಮುಂದಿನ(); }

    ತಾರ್ಕಿಕ acceptOp(ಸ್ಥಿರಾಂಕ std::string & s) {
        ಆದರೆ (tok.ಬಗೆ == ಅಭಿವ್ಯಕ್ತಿ_Lexer::OP && tok.ಪಠ್ಯ_2 == s) {
            advance();
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ತಾರ್ಕಿಕ accept_ಬಗೆ(ಅಭಿವ್ಯಕ್ತಿ_Lexer::ಬಗೆ_2 k) {
        ಆದರೆ (tok.ಬಗೆ == k) {
            advance();
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಪೂರ್ಣಾಂಕ parseLogicalOr() {
        ಪೂರ್ಣಾಂಕ v = parseLogicalAnd();
        ಆಗ (acceptOp("||")) {
            ಪೂರ್ಣಾಂಕ rhs = parseLogicalAnd();
            v       = (v || rhs);
        }
        ಹಿಂತಿರುಗಿ v;
    }

    ಪೂರ್ಣಾಂಕ parseLogicalAnd() {
        ಪೂರ್ಣಾಂಕ v = parseEquality();
        ಆಗ (acceptOp("&&")) {
            ಪೂರ್ಣಾಂಕ rhs = parseEquality();
            v       = (v && rhs);
        }
        ಹಿಂತಿರುಗಿ v;
    }

    ಪೂರ್ಣಾಂಕ parseEquality() {
        ಪೂರ್ಣಾಂಕ v = parseRelational();
        ಗಾಗಿ (;;) {
            ಆದರೆ (acceptOp("==")) {
                ಪೂರ್ಣಾಂಕ rhs = parseRelational();
                v       = (v == rhs);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (acceptOp("!=")) {
                ಪೂರ್ಣಾಂಕ rhs = parseRelational();
                v       = (v != rhs);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ನಿಲ್ಲಿಸು;
            }
        }
        ಹಿಂತಿರುಗಿ v;
    }

    ಪೂರ್ಣಾಂಕ parseRelational() {
        ಪೂರ್ಣಾಂಕ v = parse_ಸರಿಸು();
        ಗಾಗಿ (;;) {
            ಆದರೆ (acceptOp("<")) {
                ಪೂರ್ಣಾಂಕ rhs = parse_ಸರಿಸು();
                v       = (v < rhs);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (acceptOp(">")) {
                ಪೂರ್ಣಾಂಕ rhs = parse_ಸರಿಸು();
                v       = (v > rhs);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (acceptOp("<=")) {
                ಪೂರ್ಣಾಂಕ rhs = parse_ಸರಿಸು();
                v       = (v <= rhs);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (acceptOp(">=")) {
                ಪೂರ್ಣಾಂಕ rhs = parse_ಸರಿಸು();
                v       = (v >= rhs);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ನಿಲ್ಲಿಸು;
            }
        }
        ಹಿಂತಿರುಗಿ v;
    }

    ಪೂರ್ಣಾಂಕ parse_ಸರಿಸು() {
        ಪೂರ್ಣಾಂಕ v = parse_ಕೂಡಿಸು();
        ಗಾಗಿ (;;) {
            ಆದರೆ (acceptOp("<<")) {
                ಪೂರ್ಣಾಂಕ rhs = parse_ಕೂಡಿಸು();
                v       = (v << rhs);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (acceptOp(">>")) {
                ಪೂರ್ಣಾಂಕ rhs = parse_ಕೂಡಿಸು();
                v       = (v >> rhs);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ನಿಲ್ಲಿಸು;
            }
        }
        ಹಿಂತಿರುಗಿ v;
    }

    ಪೂರ್ಣಾಂಕ parse_ಕೂಡಿಸು() {
        ಪೂರ್ಣಾಂಕ v = parseMult();
        ಗಾಗಿ (;;) {
            ಆದರೆ (acceptOp("+")) {
                ಪೂರ್ಣಾಂಕ rhs = parseMult();
                v       = (v + rhs);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (acceptOp("-")) {
                ಪೂರ್ಣಾಂಕ rhs = parseMult();
                v       = (v - rhs);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ನಿಲ್ಲಿಸು;
            }
        }
        ಹಿಂತಿರುಗಿ v;
    }

    ಪೂರ್ಣಾಂಕ parseMult() {
        ಪೂರ್ಣಾಂಕ v = parseUnary();
        ಗಾಗಿ (;;) {
            ಆದರೆ (acceptOp("*")) {
                ಪೂರ್ಣಾಂಕ rhs = parseUnary();
                v       = (v * rhs);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (acceptOp("/")) {
                ಪೂರ್ಣಾಂಕ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v / rhs);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (acceptOp("%")) {
                ಪೂರ್ಣಾಂಕ rhs = parseUnary();
                v       = (rhs == 0 ? 0 : v % rhs);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ನಿಲ್ಲಿಸು;
            }
        }
        ಹಿಂತಿರುಗಿ v;
    }

    ಪೂರ್ಣಾಂಕ parseUnary() {
        ಆದರೆ (acceptOp("!")) {
            ಹಿಂತಿರುಗಿ !parseUnary();
        }
        ಆದರೆ (acceptOp("-")) {
            ಹಿಂತಿರುಗಿ -parseUnary();
        }
        ಆದರೆ (acceptOp("+")) {
            ಹಿಂತಿರುಗಿ +parseUnary();
        }
        ಹಿಂತಿರುಗಿ parsePrimary();
    }

    ಪೂರ್ಣಾಂಕ parsePrimary() {
        // '(' expr ')'
        ಆದರೆ (accept_ಬಗೆ(ಅಭಿವ್ಯಕ್ತಿ_Lexer::LPAREN)) {
            ಪೂರ್ಣಾಂಕ v = parse();
            ಆದರೆ (!accept_ಬಗೆ(ಅಭಿವ್ಯಕ್ತಿ_Lexer::RPAREN)) {
                ಎಸೆ std::runtime_error("missing ')'");
            }
            ಹಿಂತಿರುಗಿ v;
        }

        //ಸಂಖ್ಯೆ
        ಆದರೆ (tok.ಬಗೆ == ಅಭಿವ್ಯಕ್ತಿ_Lexer::ಸಂಖ್ಯೆ_2) {
            ಪೂರ್ಣಾಂಕ v = std::stoi(tok.ಪಠ್ಯ_2);
            advance();
            ಹಿಂತಿರುಗಿ v;
        }

        // defined(identifier)
        ಆದರೆ (tok.ಬಗೆ == ಅಭಿವ್ಯಕ್ತಿ_Lexer::IDENT && tok.ಪಠ್ಯ_2 == "defined") {
            advance();
            ಆದರೆ (accept_ಬಗೆ(ಅಭಿವ್ಯಕ್ತಿ_Lexer::LPAREN)) {
                ಆದರೆ (tok.ಬಗೆ != ಅಭಿವ್ಯಕ್ತಿ_Lexer::IDENT) {
                    ಎಸೆ std::runtime_error("expected identifier in defined()");
                }
                std::string name = tok.ಪಠ್ಯ_2;
                advance();
                ಆದರೆ (!accept_ಬಗೆ(ಅಭಿವ್ಯಕ್ತಿ_Lexer::RPAREN)) {
                    ಎಸೆ std::runtime_error("missing ) in defined()");
                }
                ಹಿಂತಿರುಗಿ macros.count(name) ? 1 : 0;
            } ಇಲ್ಲದಿದ್ದರೆ {
                // defined NAME
                ಆದರೆ (tok.ಬಗೆ != ಅಭಿವ್ಯಕ್ತಿ_Lexer::IDENT) {
                    ಎಸೆ std::runtime_error("expected identifier in defined NAME");
                }
                std::string name = tok.ಪಠ್ಯ_2;
                advance();
                ಹಿಂತಿರುಗಿ macros.count(name) ? 1 : 0;
            }
        }

        // identifier -> treat as integer, if defined use its value else 0
        ಆದರೆ (tok.ಬಗೆ == ಅಭಿವ್ಯಕ್ತಿ_Lexer::IDENT) {
            std::string name = tok.ಪಠ್ಯ_2;
            advance();
            ಸ್ವಯಂಚಾಲಿತ it = macros.find(name);
            ಆದರೆ (it == macros.end()) {
                ಹಿಂತಿರುಗಿ 0;
            }
            ಆದರೆ (it->second.ಖಾಲಿಗಳು()) {
                ಹಿಂತಿರುಗಿ 1;
            }
            ಹಿಂತಿರುಗಿ eval_Macro_ಅಭಿವ್ಯಕ್ತಿ(name, it->second);
        }

        // unexpected
        ಹಿಂತಿರುಗಿ 0;
    }

    ಪೂರ್ಣಾಂಕ eval_Macro_ಅಭಿವ್ಯಕ್ತಿ(ಸ್ಥಿರಾಂಕ std::string & name, ಸ್ಥಿರಾಂಕ std::string & value) {
        ಆದರೆ (visiting.count(name)) {
            ಎಸೆ std::runtime_error("Recursive macro: " + name);
        }

        visiting.insert(name);
        ಅಭಿವ್ಯಕ್ತಿ_Parser ep(value, macros, visiting);
        ಪೂರ್ಣಾಂಕ        v = ep.parse();
        visiting.erase(name);
        ಹಿಂತಿರುಗಿ v;
    }
};

//==============================================================
// Preprocessor
//==============================================================
ವರ್ಗ Preprocessor {
  ಸಾರ್ವಜನಿಕ:
    ಸ್ಪಷ್ಟ Preprocessor(Options opts = {}) : opts_(std::move(opts)) {
        // Treat empty include path as current directory
        ಆದರೆ (opts_.include_ಮಾರ್ಗ.ಖಾಲಿಗಳು()) {
            opts_.include_ಮಾರ್ಗ = ".";
        }
        parseMacroDefinitions(opts_.macros);
    }

    std::string preprocess_file(ಸ್ಥಿರಾಂಕ std::string & filename, ಸ್ಥಿರಾಂಕ std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ರಾಶಿ;
        buildMacros(additional_macros, macros, predefined);

        std::string ಫಲಿತಾಂಶ = processFile(filename, macros, predefined, include_ರಾಶಿ, Directive_ಬಹುಲಕ::All);
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    std::string preprocess(ಸ್ಥಿರಾಂಕ std::string & contents, ಸ್ಥಿರಾಂಕ std::vector<std::string> & additional_macros = {}) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ರಾಶಿ;
        buildMacros(additional_macros, macros, predefined);

        std::string ಫಲಿತಾಂಶ = processString(contents, macros, predefined, include_ರಾಶಿ, Directive_ಬಹುಲಕ::All);
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    std::string preprocess_includes_file(ಸ್ಥಿರಾಂಕ std::string & filename) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ರಾಶಿ;
        std::string ಫಲಿತಾಂಶ = processFile(filename, macros, predefined, include_ರಾಶಿ, Directive_ಬಹುಲಕ::IncludesOnly);
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

    std::string preprocess_includes(ಸ್ಥಿರಾಂಕ std::string & contents) {
        std::unordered_map<std::string, std::string> macros;
        std::unordered_set<std::string>              predefined;
        std::unordered_set<std::string>              include_ರಾಶಿ;
        std::string ಫಲಿತಾಂಶ = processString(contents, macros, predefined, include_ರಾಶಿ, Directive_ಬಹುಲಕ::IncludesOnly);
        ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
    }

  ಖಾಸಗಿ:
    Options                                      opts_;
    std::unordered_map<std::string, std::string> global_macros;

    ಎಣಿಕೆ ವರ್ಗ Directive_ಬಹುಲಕ { All, IncludesOnly };

    ರಚನೆ Cond {
        ತಾರ್ಕಿಕ ಪೋಷಕ_ಸಕ್ರಿಯ;
        ತಾರ್ಕಿಕ ಸಕ್ರಿಯ;
        ತಾರ್ಕಿಕ ತೆಗೆದುಕೊಂಡ;
    };

    //----------------------------------------------------------
    // Parse macro definitions into global_macros
    //----------------------------------------------------------
    ಶೂನ್ಯ parseMacroDefinitions(ಸ್ಥಿರಾಂಕ std::vector<std::string> & macro_defs) {
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & def : macro_defs) {
            size_t eq_pos = def.find('=');
            ಆದರೆ (eq_pos != std::string::npos) {
                // Format: NAME=VALUE
                std::string name    = trim(def.substr(0, eq_pos));
                std::string value   = trim(def.substr(eq_pos + 1));
                global_macros[name] = value;
            } ಇಲ್ಲದಿದ್ದರೆ {
                // Format: NAME
                std::string name    = trim(def);
                global_macros[name] = "";
            }
        }
    }

    //----------------------------------------------------------
    // Build combined macro map and predefined set for a preprocessing operation
    //----------------------------------------------------------
    ಶೂನ್ಯ buildMacros(ಸ್ಥಿರಾಂಕ std::vector<std::string> &               additional_macros,
                     std::unordered_map<std::string, std::string> & macros,
                     std::unordered_set<std::string> &              predefined) {
        macros = global_macros;
        predefined.clear();

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & [name, value] : global_macros) {
            predefined.insert(name);
        }

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & def : additional_macros) {
            size_t      eq_pos = def.find('=');
            std::string name, value;
            ಆದರೆ (eq_pos != std::string::npos) {
                name  = trim(def.substr(0, eq_pos));
                value = trim(def.substr(eq_pos + 1));
            } ಇಲ್ಲದಿದ್ದರೆ {
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
    std::string ತುಂಬು_File(ಸ್ಥಿರಾಂಕ std::string & fname) {
        std::ifstream f(fname);
        ಆದರೆ (!f.is_open()) {
            ಎಸೆ std::runtime_error("Could not open file: " + fname);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        ಹಿಂತಿರುಗಿ ss.str();
    }

    ತಾರ್ಕಿಕ cond_ಸಕ್ರಿಯ(ಸ್ಥಿರಾಂಕ std::vector<Cond> & cond) ಸ್ಥಿರಾಂಕ {
        ಆದರೆ (cond.ಖಾಲಿಗಳು()) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        ಹಿಂತಿರುಗಿ cond.back().ಸಕ್ರಿಯ;
    }

    //----------------------------------------------------------
    // Process a file
    //----------------------------------------------------------
    std::string processFile(ಸ್ಥಿರಾಂಕ std::string &                            name,
                            std::unordered_map<std::string, std::string> & macros,
                            ಸ್ಥಿರಾಂಕ std::unordered_set<std::string> &        predefined_macros,
                            std::unordered_set<std::string> &              include_ರಾಶಿ,
                            Directive_ಬಹುಲಕ                                  mode) {
        ಆದರೆ (include_ರಾಶಿ.count(name)) {
            ಎಸೆ std::runtime_error("Recursive include: " + name);
        }

        include_ರಾಶಿ.insert(name);
        std::string shader_ಸಂಕೇತ = ತುಂಬು_File(name);
        std::string ಹೊರಹೋಗು         = processString(shader_ಸಂಕೇತ, macros, predefined_macros, include_ರಾಶಿ, mode);
        include_ರಾಶಿ.erase(name);
        ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
    }

    std::string processIncludeFile(ಸ್ಥಿರಾಂಕ std::string &                            fname,
                                   std::unordered_map<std::string, std::string> & macros,
                                   ಸ್ಥಿರಾಂಕ std::unordered_set<std::string> &        predefined_macros,
                                   std::unordered_set<std::string> &              include_ರಾಶಿ,
                                   Directive_ಬಹುಲಕ                                  mode) {
        std::string ತುಂಬಿದ_ಮಾರ್ಗ = opts_.include_ಮಾರ್ಗ + "/" + fname;
        ಹಿಂತಿರುಗಿ processFile(ತುಂಬಿದ_ಮಾರ್ಗ, macros, predefined_macros, include_ರಾಶಿ, mode);
    }

    //----------------------------------------------------------
    // Process text
    //----------------------------------------------------------
    std::string processString(ಸ್ಥಿರಾಂಕ std::string &                            shader_ಸಂಕೇತ,
                              std::unordered_map<std::string, std::string> & macros,
                              ಸ್ಥಿರಾಂಕ std::unordered_set<std::string> &        predefined_macros,
                              std::unordered_set<std::string> &              include_ರಾಶಿ,
                              Directive_ಬಹುಲಕ                                  mode) {
        std::vector<Cond>  cond;  // Conditional stack for this shader
        std::stringstream  ಹೊರಹೋಗು;
        std::istringstream ಒಳ(shader_ಸಂಕೇತ);
        std::string        ಸಾಲು;

        ಆಗ (std::getline(ಒಳ, ಸಾಲು)) {
            std::string logical = ಸಾಲು;
            std::string t       = trim(logical);
            ಆದರೆ (!t.ಖಾಲಿಗಳು() && t[0] == '#') {
                ಆಗ (ತುದಿಗಳು_With_Continuation(logical)) {
                    stripContinuation(logical);
                    ಆದರೆ (!std::getline(ಒಳ, ಸಾಲು)) {
                        ನಿಲ್ಲಿಸು;
                    }
                    logical += "\n";
                    logical += ಸಾಲು;
                }
                t = trim(logical);
            }

            ಆದರೆ (!t.ಖಾಲಿಗಳು() && t[0] == '#') {
                ತಾರ್ಕಿಕ handled = handleDirective(t, ಹೊರಹೋಗು, macros, predefined_macros, cond, include_ರಾಶಿ, mode);
                ಆದರೆ (mode == Directive_ಬಹುಲಕ::IncludesOnly && !handled) {
                    ಹೊರಹೋಗು << logical << "\n";
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (mode == Directive_ಬಹುಲಕ::IncludesOnly) {
                    ಹೊರಹೋಗು << logical << "\n";
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (cond_ಸಕ್ರಿಯ(cond)) {
                    // Expand macros in the line before outputting
                    std::string expanded = expandMacrosRecursive(logical, macros);
                    ಹೊರಹೋಗು << expanded << "\n";
                }
            }
        }

        ಆದರೆ (mode == Directive_ಬಹುಲಕ::All && !cond.ಖಾಲಿಗಳು()) {
            ಎಸೆ std::runtime_error("Unclosed #if directive");
        }

        ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು.str();
    }

    //----------------------------------------------------------
    // Directive handler
    //----------------------------------------------------------
    ತಾರ್ಕಿಕ handleDirective(ಸ್ಥಿರಾಂಕ std::string &                            t,
                         std::stringstream &                            ಹೊರಹೋಗು,
                         std::unordered_map<std::string, std::string> & macros,
                         ಸ್ಥಿರಾಂಕ std::unordered_set<std::string> &        predefined_macros,
                         std::vector<Cond> &                            cond,
                         std::unordered_set<std::string> &              include_ರಾಶಿ,
                         Directive_ಬಹುಲಕ                                  mode) {
        // split into tokens
        std::string        body = t.substr(1);
        std::istringstream iss(body);
        std::string        cmd;
        iss >> cmd;

        ಆದರೆ (cmd == "include") {
            ಆದರೆ (mode == Directive_ಬಹುಲಕ::All && !cond_ಸಕ್ರಿಯ(cond)) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
            std::string file;
            iss >> file;
            ಆದರೆ (file.size() >= 2 && file.front() == '"' && file.back() == '"') {
                file = file.substr(1, file.size() - 2);
            }
            ಹೊರಹೋಗು << processIncludeFile(file, macros, predefined_macros, include_ರಾಶಿ, mode);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಆದರೆ (mode == Directive_ಬಹುಲಕ::IncludesOnly) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಆದರೆ (cmd == "define") {
            ಆದರೆ (!cond_ಸಕ್ರಿಯ(cond)) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
            std::string name;
            iss >> name;
            // Don't override predefined macros from options
            ಆದರೆ (predefined_macros.count(name)) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
            std::string value = trim_ಮೌಲ್ಯ(iss);
            macros[name]      = value;
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಆದರೆ (cmd == "undef") {
            ಆದರೆ (!cond_ಸಕ್ರಿಯ(cond)) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
            std::string name;
            iss >> name;
            // Don't undef predefined macros from options
            ಆದರೆ (predefined_macros.count(name)) {
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
            macros.erase(name);
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಆದರೆ (cmd == "ifdef") {
            std::string name;
            iss >> name;
            ತಾರ್ಕಿಕ p = cond_ಸಕ್ರಿಯ(cond);
            ತಾರ್ಕಿಕ v = macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಆದರೆ (cmd == "ifndef") {
            std::string name;
            iss >> name;
            ತಾರ್ಕಿಕ p = cond_ಸಕ್ರಿಯ(cond);
            ತಾರ್ಕಿಕ v = !macros.count(name);
            cond.push_back({ p, p && v, p && v });
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಆದರೆ (cmd == "if") {
            std::string ಅಭಿವ್ಯಕ್ತಿ = trim_ಮೌಲ್ಯ(iss);
            ತಾರ್ಕಿಕ        p    = cond_ಸಕ್ರಿಯ(cond);
            ತಾರ್ಕಿಕ        v    = ಸುಳ್ಳು;
            ಆದರೆ (p) {
                std::unordered_set<std::string> visiting;
                ಅಭಿವ್ಯಕ್ತಿ_Parser                      ep(ಅಭಿವ್ಯಕ್ತಿ, macros, visiting);
                v = ep.parse() != 0;
            }
            cond.push_back({ p, p && v, p && v });
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಆದರೆ (cmd == "elif") {
            std::string ಅಭಿವ್ಯಕ್ತಿ = trim_ಮೌಲ್ಯ(iss);

            ಆದರೆ (cond.ಖಾಲಿಗಳು()) {
                ಎಸೆ std::runtime_error("#elif without #if");
            }

            Cond & c = cond.back();
            ಆದರೆ (!c.ಪೋಷಕ_ಸಕ್ರಿಯ) {
                c.ಸಕ್ರಿಯ = ಸುಳ್ಳು;
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }

            ಆದರೆ (c.ತೆಗೆದುಕೊಂಡ) {
                c.ಸಕ್ರಿಯ = ಸುಳ್ಳು;
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }

            std::unordered_set<std::string> visiting;
            ಅಭಿವ್ಯಕ್ತಿ_Parser                      ep(ಅಭಿವ್ಯಕ್ತಿ, macros, visiting);
            ತಾರ್ಕಿಕ                            v = ep.parse() != 0;
            c.ಸಕ್ರಿಯ                          = v;
            ಆದರೆ (v) {
                c.ತೆಗೆದುಕೊಂಡ = ನಿಜ;
            }
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಆದರೆ (cmd == "else") {
            ಆದರೆ (cond.ಖಾಲಿಗಳು()) {
                ಎಸೆ std::runtime_error("#else without #if");
            }

            Cond & c = cond.back();
            ಆದರೆ (!c.ಪೋಷಕ_ಸಕ್ರಿಯ) {
                c.ಸಕ್ರಿಯ = ಸುಳ್ಳು;
                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
            ಆದರೆ (c.ತೆಗೆದುಕೊಂಡ) {
                c.ಸಕ್ರಿಯ = ಸುಳ್ಳು;
            } ಇಲ್ಲದಿದ್ದರೆ {
                c.ಸಕ್ರಿಯ = ನಿಜ;
                c.ತೆಗೆದುಕೊಂಡ  = ನಿಜ;
            }
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        ಆದರೆ (cmd == "endif") {
            ಆದರೆ (cond.ಖಾಲಿಗಳು()) {
                ಎಸೆ std::runtime_error("#endif without #if");
            }
            cond.pop_back();
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        // Unknown directive
        ಎಸೆ std::runtime_error("Unknown directive: #" + cmd);
    }
};

}  // namespace pre_wgsl

#ಆದರೆ_ಅಂತ್ಯ  // PRE_WGSL_HPP
