"""Generate the French PDF explaining how features predict author age."""

from fpdf import FPDF

FONTS_DIR = "C:/Windows/Fonts"


class PDF(FPDF):
    def setup_fonts(self):
        self.add_font("Arial", "", f"{FONTS_DIR}/arial.ttf")
        self.add_font("Arial", "B", f"{FONTS_DIR}/arialbd.ttf")
        self.add_font("Arial", "I", f"{FONTS_DIR}/ariali.ttf")
        self.add_font("Consolas", "", f"{FONTS_DIR}/consola.ttf")

    def header(self):
        self.set_font("Arial", "B", 10)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "TER \u2014 Pr\u00e9diction de l\u2019\u00e2ge d\u2019un auteur par CNN 1D", align="R")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Arial", "B", 16)
        self.set_text_color(30, 60, 120)
        self.cell(0, 12, title)
        self.ln(6)
        self.set_draw_color(30, 60, 120)
        self.set_line_width(0.6)
        self.line(self.get_x(), self.get_y(), self.get_x() + 190, self.get_y())
        self.ln(8)

    def sub_title(self, title):
        self.set_font("Arial", "B", 13)
        self.set_text_color(50, 50, 50)
        self.cell(0, 10, title)
        self.ln(8)

    def sub_sub_title(self, number, title):
        self.set_font("Arial", "B", 11)
        self.set_text_color(30, 60, 120)
        self.cell(0, 8, f"{number}. {title}")
        self.ln(7)

    def body_text(self, text):
        self.set_font("Arial", "", 10.5)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(3)

    def bullet(self, text):
        self.set_font("Arial", "", 10.5)
        self.set_text_color(40, 40, 40)
        self.cell(8, 6, "  \u2022")
        self.multi_cell(0, 6, text)
        self.ln(1)

    def code_block(self, text):
        self.set_font("Consolas", "", 9.5)
        self.set_text_color(60, 60, 60)
        self.set_fill_color(240, 240, 245)
        for line in text.split("\n"):
            self.set_x(self.get_x() + 5)
            self.cell(170, 5.5, line, fill=True)
            self.ln(5.5)
        self.ln(4)

    def table_row(self, cells, header=False):
        style = "B" if header else ""
        if header:
            self.set_fill_color(30, 60, 120)
            self.set_text_color(255, 255, 255)
        else:
            self.set_fill_color(245, 245, 250)
            self.set_text_color(40, 40, 40)
        self.set_font("Arial", style, 10)
        col_widths = [45, 75, 70]
        for i, cell in enumerate(cells):
            self.cell(col_widths[i], 8, cell, border=1, fill=True, align="C")
        self.ln()


def build_pdf():
    pdf = PDF()
    pdf.setup_fonts()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── PAGE 1 ─────────────────────────────────────────────────────────────
    pdf.add_page()

    pdf.set_font("Arial", "B", 22)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 15, "Comment les caract\u00e9ristiques", align="C")
    pdf.ln(12)
    pdf.cell(0, 15, "stylistiques pr\u00e9disent l\u2019\u00e2ge", align="C")
    pdf.ln(20)

    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "Projet TER \u2014 CNN Temporel 1D pour la pr\u00e9diction", align="C")
    pdf.ln(7)
    pdf.cell(0, 8, "de l\u2019\u00e2ge d\u2019auteurs fran\u00e7ais (30\u201370 ans)", align="C")
    pdf.ln(25)

    # ── SECTION 1 ──────────────────────────────────────────────────────────
    pdf.section_title("1. L\u2019id\u00e9e fondamentale")

    pdf.body_text(
        "Notre mod\u00e8le ne cherche pas \u00e0 comprendre de quoi parle un auteur. "
        "Il d\u00e9tecte comment l\u2019auteur \u00e9crit \u2014 son empreinte stylistique. "
        "Cette empreinte \u00e9volue avec l\u2019\u00e2ge : le vocabulaire, la complexit\u00e9 "
        "des phrases, les habitudes de ponctuation, la morphologie des mots "
        "changent au fil des ann\u00e9es."
    )
    pdf.body_text(
        "Nous extrayons 14 caract\u00e9ristiques num\u00e9riques pour chaque mot du "
        "texte. Ces caract\u00e9ristiques ne portent aucune information s\u00e9mantique "
        "(le sens des mots) \u2014 elles capturent uniquement la forme, la "
        "structure et le rythme de l\u2019\u00e9criture."
    )

    # ── SECTION 2 ──────────────────────────────────────────────────────────
    pdf.section_title("2. Les 14 caract\u00e9ristiques et leur lien avec l\u2019\u00e2ge")

    # -- Frequency --
    pdf.sub_sub_title("2.1", "Fr\u00e9quence (4 caract\u00e9ristiques)")
    pdf.body_text("freq_in_text \u2014 log_freq \u2014 freq_rank \u2014 global_freq")
    pdf.body_text(
        "Ces quatre valeurs mesurent \u00e0 quel point un mot est r\u00e9p\u00e9t\u00e9 dans le "
        "texte (comptage brut, \u00e9chelle logarithmique, rang) et sa fr\u00e9quence "
        "globale dans l\u2019ensemble du corpus d\u2019entra\u00eenement."
    )
    pdf.bullet(
        "Les auteurs plus \u00e2g\u00e9s tendent \u00e0 utiliser un vocabulaire plus "
        "stable et consolid\u00e9, avec davantage de r\u00e9p\u00e9titions de mots "
        "familiers."
    )
    pdf.bullet(
        "Les auteurs plus jeunes emploient souvent un vocabulaire plus "
        "diversifi\u00e9, avec plus de mots rares et moins de r\u00e9p\u00e9titions."
    )
    pdf.body_text(
        "La distribution des fr\u00e9quences sur un texte entier est un signal "
        "stylistique fort, bien document\u00e9 en stylom\u00e9trie."
    )

    # -- Word length --
    pdf.sub_sub_title("2.2", "Longueur des mots (2 caract\u00e9ristiques)")
    pdf.body_text("word_length \u2014 syllable_count")
    pdf.bullet(
        "Les auteurs plus \u00e2g\u00e9s tendent \u00e0 utiliser des mots plus longs et "
        "plus complexes \u2014 leur vocabulaire acquis est plus vaste."
    )
    pdf.bullet(
        "Les auteurs plus jeunes privil\u00e9gient souvent des mots plus "
        "courts et plus directs."
    )
    pdf.body_text(
        "Le nombre moyen de syllabes par mot est une m\u00e9trique classique de "
        "lisibilit\u00e9 et de complexit\u00e9 textuelle (indice de Flesch, etc.)."
    )

    # -- Char composition --
    pdf.sub_sub_title("2.3", "Composition des caract\u00e8res (2 caract\u00e9ristiques)")
    pdf.body_text("vowel_ratio \u2014 accent_type")
    pdf.bullet(
        "Les mots accentu\u00e9s en fran\u00e7ais (\u00e9, \u00e8, \u00ea, \u00e7\u2026) signalent souvent "
        "un registre plus formel ou litt\u00e9raire. Les auteurs plus \u00e2g\u00e9s "
        "tendent vers un registre plus soutenu."
    )
    pdf.bullet(
        "Le ratio de voyelles capture des patterns phonologiques qui "
        "varient selon le style d\u2019\u00e9criture."
    )

    # -- Lexical --
    pdf.sub_sub_title("2.4", "Caract\u00e9ristiques lexicales (2 caract\u00e9ristiques)")
    pdf.body_text("punctuation_type \u2014 pos_tag")
    pdf.bullet(
        "Ponctuation : les points-virgules, les deux-points et les "
        "tirets sont plus fr\u00e9quents dans l\u2019\u00e9criture mature. Les jeunes "
        "auteurs \u00e9crivent des phrases plus courtes avec des points."
    )
    pdf.bullet(
        "Cat\u00e9gorie grammaticale (POS tag) : gr\u00e2ce au mod\u00e8le spaCy fran\u00e7ais, "
        "chaque mot re\u00e7oit un code identifiant sa cat\u00e9gorie (adjectif, nom "
        "commun, nom propre, verbe, adverbe, d\u00e9terminant, pronom\u2026). La "
        "distribution des cat\u00e9gories grammaticales varie avec l\u2019\u00e2ge et le "
        "niveau de ma\u00eetrise linguistique."
    )

    # -- Positional --
    pdf.sub_sub_title("2.5", "Caract\u00e9ristiques positionnelles (3 caract\u00e9ristiques)")
    pdf.body_text("pos_in_sentence \u2014 sentence_length \u2014 is_sentence_boundary")
    pdf.bullet(
        "La longueur des phrases est un indicateur majeur de l\u2019\u00e2ge : les "
        "auteurs plus \u00e2g\u00e9s \u00e9crivent des phrases plus longues et plus "
        "structur\u00e9es."
    )
    pdf.bullet(
        "Les patterns de fronti\u00e8res de phrase r\u00e9v\u00e8lent le rythme et les "
        "pr\u00e9f\u00e9rences de cadence de l\u2019auteur."
    )

    # -- Context --
    pdf.sub_sub_title("2.6", "Caract\u00e9ristiques de contexte (1 caract\u00e9ristique)")
    pdf.body_text("adjacent_to_period")
    pdf.bullet(
        "Un mot imm\u00e9diatement voisin d\u2019un point final ( . ) : le jeton "
        "pr\u00e9c\u00e9dent ou le suivant est un point. Apr\u00e8s un point, on est en "
        "d\u00e9but de phrase ; avant un point, en fin de phrase. Ce signal est "
        "distinct de is_sentence_boundary (qui couvre aussi ! et ?) et ne "
        "concerne que le point."
    )

    # ── SECTION 3 ──────────────────────────────────────────────────────────
    pdf.section_title("3. Le r\u00f4le du CNN : apprendre les combinaisons")

    pdf.body_text(
        "Point essentiel : nous ne programmons jamais manuellement de "
        "r\u00e8gles du type \u00ab si les phrases sont longues, alors auteur \u00e2g\u00e9 \u00bb. "
        "Aucune caract\u00e9ristique isol\u00e9e ne suffit \u00e0 pr\u00e9dire l\u2019\u00e2ge. C\u2019est la "
        "combinaison et l\u2019encha\u00eenement de ces caract\u00e9ristiques qui porte "
        "l\u2019information."
    )

    pdf.sub_title("Le parcours des donn\u00e9es")

    pdf.body_text("\u00c9tape 1 \u2014 Extraction")
    pdf.body_text(
        "Pour chaque mot du texte, on calcule les 14 caract\u00e9ristiques. "
        "Le texte devient une matrice (1000 mots x 14 valeurs)."
    )

    pdf.body_text("\u00c9tape 2 \u2014 D\u00e9tection de motifs (3 branches parall\u00e8les)")
    pdf.body_text(
        "Le r\u00e9seau utilise 3 branches parall\u00e8les, chacune sp\u00e9cialis\u00e9e "
        "dans une \u00e9chelle de motifs diff\u00e9rente. Chaque branche empile "
        "3 couches de Conv1d + MaxPool(\u00f72) :"
    )
    pdf.bullet(
        "Branche k=3 : 3 couches [Conv1d(k=3) \u2192 MaxPool(2)] \u2014 capture "
        "les motifs locaux (bigrammes/trigrammes, ex : \u00ab ne...pas \u00bb)"
    )
    pdf.bullet(
        "Branche k=7 : 3 couches [Conv1d(k=7) \u2192 MaxPool(2)] \u2014 d\u00e9tecte "
        "les motifs au niveau du syntagme (tournures de phrase)"
    )
    pdf.bullet(
        "Branche k=13 : 3 couches [Conv1d(k=13) \u2192 MaxPool(2)] \u2014 rep\u00e8re "
        "les structures au niveau de la proposition et de la phrase"
    )
    pdf.body_text(
        "\u00c0 chaque couche, le MaxPool r\u00e9duit la s\u00e9quence de moiti\u00e9 en ne "
        "gardant que la valeur la plus forte sur chaque paire de positions "
        "cons\u00e9cutives. Apr\u00e8s 3 couches, la s\u00e9quence est r\u00e9duite \u00e0 environ "
        "1/8 de sa taille initiale."
    )
    pdf.code_block(
        "Entr\u00e9e :  1000 mots \u00d7 14 features\n"
        "\n"
        "Branche k=3 :  1000 \u2192 499 \u2192 248 \u2192 123\n"
        "Branche k=7 :  1000 \u2192 497 \u2192 245 \u2192 119\n"
        "Branche k=13 : 1000 \u2192 494 \u2192 241 \u2192 114"
    )

    pdf.body_text("\u00c9tape 3 \u2014 Agr\u00e9gation et concat\u00e9nation")
    pdf.body_text(
        "Un Global Average Pooling r\u00e9duit chaque branche \u00e0 un vecteur de "
        "64 valeurs. Les 3 vecteurs sont concat\u00e9n\u00e9s : 64 + 64 + 64 = 192 "
        "valeurs r\u00e9sumant les motifs d\u00e9tect\u00e9s \u00e0 toutes les \u00e9chelles."
    )

    pdf.body_text("\u00c9tape 4 \u2014 Classification (couche lin\u00e9aire)")
    pdf.body_text(
        "Les 192 valeurs (64 filtres \u00d7 3 branches) sont combin\u00e9es par "
        "une couche enti\u00e8rement connect\u00e9e qui produit 41 scores \u2014 un par "
        "\u00e2ge de 30 \u00e0 70 ans. L\u2019\u00e2ge avec le score le plus \u00e9lev\u00e9 est la "
        "pr\u00e9diction."
    )

    # ── SECTION 4 ──────────────────────────────────────────────────────────
    pdf.section_title("4. Analogie")

    pdf.body_text(
        "Imaginons un m\u00e9decin qui estime l\u2019\u00e2ge d\u2019un patient \u00e0 partir "
        "d\u2019une prise de sang :"
    )
    pdf.bullet(
        "Les caract\u00e9ristiques sont comme les marqueurs sanguins "
        "(cholest\u00e9rol, glyc\u00e9mie, cr\u00e9atinine...). Chacun est une mesure "
        "objective et interpr\u00e9table."
    )
    pdf.bullet(
        "Le CNN est comme le m\u00e9decin qui a vu des milliers de patients "
        "et qui a appris quelles combinaisons de marqueurs correspondent "
        "\u00e0 quelle tranche d\u2019\u00e2ge."
    )
    pdf.bullet(
        "Aucun marqueur isol\u00e9 ne d\u00e9termine l\u2019\u00e2ge \u2014 c\u2019est le pattern "
        "d\u2019ensemble qui compte. Le r\u00e9seau d\u00e9couvre ces patterns "
        "automatiquement pendant l\u2019entra\u00eenement."
    )

    # ── SECTION 5 ──────────────────────────────────────────────────────────
    pdf.section_title("5. R\u00e9capitulatif des 14 caract\u00e9ristiques")

    pdf.table_row(["Groupe", "Caract\u00e9ristiques", "Dimension"], header=True)
    pdf.table_row(["Fr\u00e9quence", "freq, log_freq, rank, global", "4"])
    pdf.table_row(["Longueur", "word_length, syllabes", "2"])
    pdf.table_row(["Composition", "vowel_ratio, accent", "2"])
    pdf.table_row(["Lexical", "punct_type, pos_tag", "2"])
    pdf.table_row(["Positionnel", "pos, sent_len, boundary", "3"])
    pdf.table_row(["Contexte", "adjacent_to_period", "1"])
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(30, 60, 120)
    pdf.cell(120, 8, "", border=0)
    pdf.cell(70, 8, "Total : 14", border=0)
    pdf.ln(15)

    pdf.body_text(
        "Ces 14 valeurs par mot, combin\u00e9es sur des fen\u00eatres de 1000 mots "
        "et analys\u00e9es par le CNN multi-branches, permettent au mod\u00e8le de "
        "capturer l\u2019empreinte stylistique propre \u00e0 chaque tranche d\u2019\u00e2ge."
    )

    # ── Save ───────────────────────────────────────────────────────────────
    output_path = "assets/caracteristiques_et_age.pdf"
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")


if __name__ == "__main__":
    build_pdf()
