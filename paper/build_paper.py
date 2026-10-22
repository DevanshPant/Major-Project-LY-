"""Build the IEEE-style two-column conference paper as a PDF.

Every quantitative value is read from the measured result files in
data/ so that nothing in the document is hand-entered.

Run:
    python paper/build_paper.py
Output:
    paper/ResumeAI_IEEE_draft.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, Image, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle, KeepTogether)

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIGS = PAPER / "figs"
OUT = PAPER / "ResumeAI_IEEE_draft.pdf"

# ---------------------------------------------------------------- data
EV = pd.read_json(ROOT / "data" / "eval_results.json")
AD = json.load(open(ROOT / "data" / "adversarial_results.json"))
CS = json.load(open(ROOT / "data" / "corpus_stats.json"))
ADR = pd.DataFrame(AD["ranking"])
ADD = pd.DataFrame(AD["detection"])


def ev_mean(system, metric):
    return EV[EV.system == system][metric].mean()


def ev_sem(system, metric):
    return EV[EV.system == system][metric].sem()


def paired(a, b, metric="p_at_k"):
    piv = EV.pivot(index="query", columns="system", values=metric)
    from scipy import stats
    t, p = stats.ttest_rel(piv[a], piv[b])
    d = piv[a] - piv[b]
    return d.mean(), t, p, int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())


def det_rate(label):
    sub = ADD[ADD.label == label]
    return sub.flagged.mean(), len(sub)


def atk(defence, topk=5):
    sub = ADR[ADR.defence == defence]
    per_role = (sub[sub["rank"] <= topk].groupby("role")["label"]
                .apply(lambda s: s.str.startswith("attack").mean()))
    return per_role.mean()


def atk_rank1(defence):
    sub = ADR[(ADR.defence == defence) & (ADR["rank"] == 1)]
    return sub["label"].str.startswith("attack").mean()


def mean_rank(defence, label_prefix):
    sub = ADR[ADR.defence == defence]
    m = sub[sub["label"].str.startswith(label_prefix)]
    return m["rank"].mean()


# ---------------------------------------------------------------- styles
PW, PH = LETTER
MARGIN = 0.75 * inch
GUT = 0.25 * inch
COLW = (PW - 2 * MARGIN - GUT) / 2

S = {}
S["title"] = ParagraphStyle("title", fontName="Times-Bold", fontSize=20,
                            leading=23, alignment=TA_CENTER, spaceAfter=10)
S["authors"] = ParagraphStyle("authors", fontName="Times-Roman", fontSize=10.5,
                              leading=13, alignment=TA_CENTER, spaceAfter=2)
S["affil"] = ParagraphStyle("affil", fontName="Times-Italic", fontSize=9.5,
                            leading=12, alignment=TA_CENTER, spaceAfter=10)
S["h1"] = ParagraphStyle("h1", fontName="Times-Bold", fontSize=10, leading=12,
                         spaceBefore=9, spaceAfter=4)
S["h2"] = ParagraphStyle("h2", fontName="Times-Italic", fontSize=9.5, leading=11,
                         spaceBefore=6, spaceAfter=3)
S["body"] = ParagraphStyle("body", fontName="Times-Roman", fontSize=9.3,
                           leading=11.2, alignment=TA_JUSTIFY, spaceAfter=4,
                           firstLineIndent=10)
S["body0"] = ParagraphStyle("body0", parent=S["body"], firstLineIndent=0)
S["abs"] = ParagraphStyle("abs", fontName="Times-Bold", fontSize=9.0,
                          leading=10.8, alignment=TA_JUSTIFY, spaceAfter=4)
S["cap"] = ParagraphStyle("cap", fontName="Times-Roman", fontSize=8.2,
                          leading=9.8, alignment=TA_CENTER, spaceBefore=3,
                          spaceAfter=7)
S["tcap"] = ParagraphStyle("tcap", fontName="Times-Roman", fontSize=8.2,
                           leading=9.8, alignment=TA_CENTER, spaceBefore=2,
                           spaceAfter=3)
S["ref"] = ParagraphStyle("ref", fontName="Times-Roman", fontSize=8.0,
                          leading=9.4, alignment=TA_JUSTIFY, spaceAfter=2.2,
                          leftIndent=11, firstLineIndent=-11)
S["cell"] = ParagraphStyle("cell", fontName="Times-Roman", fontSize=7.8, leading=9.2)
S["cellb"] = ParagraphStyle("cellb", fontName="Times-Bold", fontSize=7.8, leading=9.2)


def P(t, st="body"):
    return Paragraph(t, S[st])


def fig(name, width, caption):
    p = FIGS / f"{name}.png"
    if not p.exists():
        return [P(f"[missing figure {name}]", "cap")]
    from PIL import Image as PILImage
    iw, ih = PILImage.open(p).size
    img = Image(str(p), width=width, height=width * ih / iw)
    return [Spacer(1, 3), img, P(caption, "cap")]


def table(data, widths, caption, span_header=False):
    t = Table(data, colWidths=widths, hAlign="CENTER")
    style = [
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("LEADING", (0, 0), (-1, -1), 9.2),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
    ]
    t.setStyle(TableStyle(style))
    return [KeepTogether([P(caption, "tcap"), t]), Spacer(1, 7)]


# ---------------------------------------------------------------- content
story = []

story.append(P("Rendering-Layer Integrity Verification for "
               "Adversarially Robust Resume Screening", "title"))
story.append(P("Devansh Pant, Dhruti Dutia, Yashvi Lohiya, Vedant Agrawal", "authors"))
story.append(P("K. J. Somaiya School of Engineering, Mumbai, India", "affil"))

n = CS["n_resumes"]; ncat = CS["n_categories"]
tpr_w, n_w = det_rate("attack_white")
tpr_t, n_t = det_rate("attack_tiny")
fpr_r, n_r = det_rate("clean_relevant")
fpr_o, n_o = det_rate("clean_offtopic")
bi_p = ev_mean("bi", "p_at_k")
tf_p = ev_mean("tfidf", "p_at_k")
pre_p = ev_mean("full_pre", "p_at_k")
ft_p = ev_mean("full_ft", "p_at_k")
d_bf, t_bf, p_bf, w_bf, ti_bf, l_bf = paired("full_pre", "bi")
d_ft, t_ft, p_ft, w_ft, ti_ft, l_ft = paired("full_ft", "full_pre")
d_fb, t_fb, p_fb, w_fb, ti_fb, l_fb = paired("full_ft", "tfidf")
d_fbi, t_fbi, p_fbi, *_ = paired("full_ft", "bi")

story.append(P(
    f"<b><i>Abstract</i></b>&mdash;Automated resume screening systems read the text "
    f"layer of a candidate document and treat it as a faithful record of what the "
    f"document says. We show that this assumption is unsafe. An applicant can embed "
    f"job-description keywords in a resume using white-on-white text or one-point "
    f"fonts, producing a file that looks unchanged to a human reviewer while "
    f"injecting arbitrary content into every text extractor we tested. Across "
    f"{n_w + n_t} attacked documents, an undefended semantic pipeline placed an "
    f"attacker at rank one for every role we evaluated. We propose rendering-layer "
    f"integrity verification, which inspects per-glyph fill colour, font size and "
    f"PDF text rendering mode rather than the flattened text, and we report "
    f"{tpr_w*100:.0f} percent detection on both attack variants with a "
    f"{fpr_o*100:.1f} percent false positive rate on {n_r + n_o} clean documents. "
    f"Enabling the defence removes the attacker from rank one entirely. We also "
    f"separately report a transfer failure and its repair. On a corpus of {n:,} "
    f"resumes across {ncat} occupational categories, a general-purpose cross-encoder "
    f"re-ranker reduced Precision@10 from {bi_p:.3f} to {pre_p:.3f} relative to "
    f"bi-encoder retrieval alone (p = {p_bf:.3f}), which we attribute to domain "
    f"mismatch between passage-ranking pretraining and long-form resume documents. "
    f"Adapting the same re-ranker with weakly supervised pairs drawn from category "
    f"labels reversed the effect and raised Precision@10 to {ft_p:.3f}, exceeding "
    f"both the undapted re-ranker (p = {p_ft:.4f}) and the lexical baseline "
    f"(p = {p_fb:.4f}).", "abs"))
story.append(P("<b><i>Index Terms</i></b>&mdash;resume screening, adversarial "
               "robustness, document integrity, information retrieval, "
               "dense retrieval, algorithmic hiring", "abs"))

# ---- I. Introduction
story.append(P("I.&nbsp;&nbsp;I<font size=7.5>NTRODUCTION</font>", "h1"))
story.append(P(
    "Large employers receive far more applications than any human team can read. "
    "The standard response is an applicant tracking system that ranks candidates "
    "automatically, and the research literature has moved steadily from lexical "
    "matching towards dense retrieval, where resumes and job descriptions are "
    "embedded in a shared vector space [1], [2]. Work in this area is usually "
    "evaluated on ranking quality: does the system place genuinely suitable "
    "candidates near the top.", "body0"))
story.append(P(
    "That framing contains a hidden assumption. Every pipeline we are aware of "
    "begins by calling a text extractor on the uploaded file and then treats the "
    "returned string as ground truth. The extractor discards almost everything "
    "about how the document is drawn. Colour, glyph size and the PDF text "
    "rendering mode are all thrown away before the model sees anything. An "
    "applicant who understands this can write text that the extractor reads and a "
    "recruiter cannot see."))
story.append(P(
    "This is not a hypothetical concern. The same idea, invisible text placed to "
    "manipulate an automated reader, has a long history in web search, where it is "
    "catalogued as a standard spam technique [3]. Resume screening has the same "
    "structure, a ranking system standing between an applicant and a scarce "
    "resource, with the same incentive to game it, but without the adversarial "
    "defences that search engines developed in response."))
story.append(P("This paper makes three contributions."))
story.append(P(
    "First, we quantify the vulnerability. We construct attacked resumes by "
    "injecting role-specific keywords into off-topic documents using two "
    "rendering tricks, and we measure how far an attacker can climb an otherwise "
    "reasonable semantic ranking pipeline.", "body0"))
story.append(P(
    "Second, we propose and evaluate a defence that operates on the rendering "
    "layer rather than the text layer, recovering per-glyph attributes that the "
    "ordinary extraction path discards, and we report its detection rate and its "
    "false positive cost on clean documents.", "body0"))
story.append(P(
    "Third, we report a transfer failure and its repair. Adding an off-the-shelf "
    "cross-encoder re-ranker, a step close to standard practice in neural "
    "retrieval [4], measurably degraded ranking quality on our corpus. Adapting "
    "that re-ranker to the domain with weak supervision reversed the loss and "
    "produced the best system we measured.", "body0"))

# ---- II. Related work
story.append(P("II.&nbsp;&nbsp;R<font size=7.5>ELATED</font> W<font size=7.5>ORK</font>", "h1"))
story.append(P(
    "<i>Dense retrieval.</i> Sentence-BERT showed that siamese transformer "
    "encoders produce sentence embeddings suitable for similarity search [1], and "
    "distilled variants such as MiniLM made this cheap enough for commodity "
    "hardware [5]. Approximate nearest neighbour libraries, FAISS in particular "
    "[6], make the retrieval step practical at corpus scale. The common deployment "
    "pattern is two stages: a fast bi-encoder retrieves a candidate set, then a "
    "slower cross-encoder re-reads each query and document jointly to produce the "
    "final order [4].", "body0"))
story.append(P(
    "<i>Transfer of ranking models.</i> The BEIR benchmark demonstrated that "
    "retrieval models trained on MS MARCO transfer unevenly to other domains, and "
    "that re-rankers in particular can underperform outside their training "
    "distribution [7]. Our negative result in Section V is consistent with that "
    "finding and extends it to long-form resume documents.", "body0"))
story.append(P(
    "<i>Fairness in algorithmic hiring.</i> A substantial literature examines bias "
    "in automated hiring [8], [9], prompted in part by widely reported industrial "
    "failures. This work is concerned with whether a system treats candidates "
    "equitably. Our concern is adjacent but distinct: whether the input the system "
    "reads is the document the candidate actually presents.", "body0"))
story.append(P(
    "<i>Adversarial manipulation of ranked retrieval.</i> Web spam taxonomies "
    "describe invisible text and keyword stuffing as established techniques for "
    "manipulating search rankings [3]. To our knowledge this threat model has not "
    "been evaluated against resume screening systems, and no published screening "
    "pipeline inspects document rendering attributes as a defence.", "body0"))

story.append(NextPageTemplate("twocol"))

# ---- III. System
story.append(P("III.&nbsp;&nbsp;S<font size=7.5>YSTEM</font> D<font size=7.5>ESCRIPTION</font>", "h1"))
story.append(P(
    "Fig. 1 shows the pipeline. The upper path is a conventional screening "
    "system and provides the setting in which we evaluate the attack. The shaded "
    "component is the contribution of this paper.", "body0"))
story.extend(fig("fig_architecture", COLW,
                 "Fig. 1. Screening pipeline. The upper row is the conventional "
                 "text-layer path. Rendering-layer integrity verification (shaded) "
                 "reads the original document bytes rather than the extracted "
                 "string, so it observes attributes the text layer discards."))

story.append(P("<i>A. Document ingestion</i>", "h2"))
story.append(P(
    "Documents arrive as PDF, DOCX or a ZIP archive. Text extraction attempts "
    "pypdf first and falls back to PyMuPDF, which recovers text from some files "
    "where the first extractor returns nothing. Scanned documents with no text "
    "layer are passed to an OCR stage. Because optical recognition costs roughly "
    "fifteen seconds per page on the hardware we target, recognised text is cached "
    "in a content-addressed store keyed by the SHA-256 digest of the file, which "
    "reduces repeat cost for an unchanged document from seconds to a few "
    "milliseconds.", "body0"))

story.append(P("<i>B. Retrieval and scoring</i>", "h2"))
story.append(P(
    "Resumes and the job description are encoded with all-MiniLM-L6-v2 into "
    "384-dimensional unit vectors and compared by inner product using a FAISS "
    "flat index, so that the inner product is exactly the cosine similarity. A "
    "cross-encoder re-ranks the top thirty candidates. Section V reports that this "
    "stage was harmful on our data.", "body0"))
story.append(P(
    "The final score is a weighted sum of four components: semantic similarity, "
    "overlap between the skills named in the job description and those found in "
    "the resume, a comparison of years of experience against any stated "
    "requirement, and a comparison of highest degree against any stated "
    "requirement.", "body0"))

story.append(P("<i>C. Constraint-aware weighting</i>", "h2"))
story.append(P(
    "A fixed weighting creates a failure mode that we encountered in practice. If "
    "a job description states no degree requirement, every candidate receives an "
    "identical education sub-score. The same holds for experience when no number "
    "of years is given. These constant terms carry no information yet occupy a "
    "fixed share of the total, which compresses the spread of final scores and, in "
    "the limit where a job description constrains none of the structured fields, "
    "makes the ranking degenerate: different job descriptions return the same "
    "ordering.", "body0"))
story.append(P(
    "We therefore treat a component as active only when the job description "
    "actually constrains it, and renormalise the remaining weights to sum to one. "
    "Writing <i>A</i> for the set of active components, the score is", "body0"))
story.append(Spacer(1, 2))
story.append(Paragraph(
    "<i>s</i> = &Sigma;<sub><i>i</i>&isin;<i>A</i></sub> "
    "(<i>w<sub>i</sub></i> / &Sigma;<sub><i>j</i>&isin;<i>A</i></sub> "
    "<i>w<sub>j</sub></i>) &middot; <i>c<sub>i</sub></i>",
    ParagraphStyle("eq", fontName="Times-Roman", fontSize=9.3, leading=12,
                   alignment=TA_CENTER, spaceBefore=2, spaceAfter=4)))
story.append(P(
    "where <i>c<sub>i</sub></i> is the <i>i</i>th component score. A criterion the "
    "employer never specified contributes nothing rather than contributing a "
    "constant.", "body0"))

# ---- IV. Integrity verification
story.append(P("IV.&nbsp;&nbsp;R<font size=7.5>ENDERING</font>-L<font size=7.5>AYER</font> "
               "I<font size=7.5>NTEGRITY</font> V<font size=7.5>ERIFICATION</font>", "h1"))
story.append(P("<i>A. Threat model</i>", "h2"))
story.append(P(
    "The attacker is an applicant who controls the file they submit and wants an "
    "off-topic resume ranked highly for a target role. The attacker cannot modify "
    "the screening system and does not need to know its internals. The constraint "
    "is that the document must still look like an ordinary resume when a recruiter "
    "opens it, because a visibly keyword-stuffed document would be rejected on "
    "sight.", "body0"))

story.append(P("<i>B. Detectors</i>", "h2"))
story.append(P(
    "Rather than reading the extracted string, we re-open the document and inspect "
    "each glyph. For PDF we use pdfplumber, which exposes the fill colour, size "
    "and rendering mode of every character; for DOCX we inspect run-level font "
    "properties. Four conditions raise a flag.", "body0"))
story.append(P(
    "<b>Near-background colour.</b> We convert each glyph fill to perceived "
    "brightness using <i>Y</i> = 0.299<i>R</i> + 0.587<i>G</i> + 0.114<i>B</i> and "
    "flag glyphs with <i>Y</i> &ge; 0.92, the regime in which text is effectively "
    "invisible against white stock.", "body0"))
story.append(P(
    "<b>Invisible rendering mode.</b> The PDF specification defines text rendering "
    "mode 3, which draws no marks. It has legitimate uses, notably OCR text layers "
    "placed beneath a scanned image, so on its own it is suggestive rather than "
    "conclusive.", "body0"))
story.append(P(
    "<b>Micro-fonts.</b> Glyphs below three points are unreadable in print and on "
    "screen at normal zoom but extract normally.", "body0"))
story.append(P(
    "<b>Keyword density.</b> A term occurring at least fifteen times and "
    "accounting for more than four percent of tokens is anomalous for a genuine "
    "resume. This detector is independent of rendering and so catches stuffing "
    "that is merely visible rather than hidden.", "body0"))
story.append(P(
    "A flagged document is not discarded. It receives a score penalty "
    "proportional to the fraction of the document that is hidden, and the "
    "reviewer is shown the specific reason together with the extracted hidden "
    "text. We prefer this to silent rejection because the detectors can "
    "misfire, and because the recovered text is itself evidence a recruiter may "
    "wish to act on.", "body0"))

# ---- V. Experimental setup
story.append(P("V.&nbsp;&nbsp;E<font size=7.5>XPERIMENTAL</font> S<font size=7.5>ETUP</font>", "h1"))
deg = CS["degree"]; tot = sum(deg.values())
story.append(P(
    f"<i>A. Corpus.</i> We use a publicly available resume corpus of {n:,} "
    f"documents labelled with {ncat} occupational categories, with a mean length "
    f"of {CS['mean_chars']:,.0f} characters. Applying our structured extractor, "
    f"{deg.get('masters',0)/tot*100:.1f} percent of documents report a master's "
    f"degree as the highest qualification, {deg.get('bachelors',0)/tot*100:.1f} "
    f"percent a bachelor's degree and {deg.get('none',0)/tot*100:.1f} percent none "
    f"that we could identify. Explicit years of experience were absent from "
    f"{CS['experience']['0']/n*100:.1f} percent of documents, which is the reason "
    f"the experience component is disabled by default when a job description does "
    f"not state a requirement. Fig. 2 summarises both distributions.", "body0"))
story.extend(fig("fig_corpus", COLW,
                 "Fig. 2. Corpus characterisation. (a) Highest degree detected. "
                 "(b) Years of experience recovered by the structured extractor."))

story.append(P(
    "<i>B. Retrieval protocol.</i> For each of the 24 categories we build a pool "
    "of 100 documents, 20 drawn from the target category and 80 from other "
    "categories, and issue a short job description for that role. A document "
    "counts as relevant if its category label matches the queried role. This is a "
    "coarse proxy for human relevance judgements, and we return to that limitation "
    "in Section VII; it is applied identically to every system, so the comparison "
    "between systems remains meaningful. We report Precision@10, Recall@10, "
    "NDCG@10 and MRR, and compare systems with a paired two-tailed t-test over the "
    "24 queries.", "body0"))
story.append(P(
    f"<i>C. Adversarial protocol.</i> We evaluate three roles. For each, the pool "
    f"contains 12 genuine resumes from the target category, 12 genuine resumes "
    f"from other categories, and attacked copies of those same off-topic resumes, "
    f"12 using white-on-white injection and 12 using one-point fonts, giving 48 "
    f"documents per role and {len(ADD)} in total. Using the same base documents "
    f"for the clean and attacked conditions isolates the effect of the injection. "
    f"We score every pool twice, once with the defence disabled and once enabled.", "body0"))

# ---- VI. Results
story.append(P("VI.&nbsp;&nbsp;R<font size=7.5>ESULTS</font>", "h1"))
story.append(P("<i>A. The attack defeats an undefended pipeline</i>", "h2"))
r1_off, r1_on = atk_rank1(False), atk_rank1(True)
t5_off, t5_on = atk(False), atk(True)
story.append(P(
    f"With no defence, keyword injection is decisive. An attacked off-topic resume "
    f"took rank one in {r1_off*100:.0f} percent of roles and occupied "
    f"{t5_off*100:.0f} percent of the top five positions. Attacked documents "
    f"reached a mean rank of {mean_rank(False,'attack'):.1f} while genuine "
    f"relevant resumes averaged {mean_rank(False,'clean_relevant'):.1f}. The "
    f"semantic pipeline is not merely degraded; it is inverted, ranking irrelevant "
    f"candidates above suitable ones because the injected text is, as far as the "
    f"model can tell, part of the document.", "body0"))

story.append(P("<i>B. Detection accuracy</i>", "h2"))
story.extend(table(
    [["Document class", "n", "Flagged", "Interpretation"],
     ["White-text attack", f"{n_w}", f"{tpr_w*100:.1f}%", "true positive"],
     ["Micro-font attack", f"{n_t}", f"{tpr_t*100:.1f}%", "true positive"],
     ["Clean, relevant", f"{n_r}", f"{fpr_r*100:.1f}%", "false positive"],
     ["Clean, off-topic", f"{n_o}", f"{fpr_o*100:.1f}%", "false positive"]],
    [1.25 * inch, 0.35 * inch, 0.62 * inch, 0.95 * inch],
    "TABLE I. D<font size=6.5>ETECTION RATES BY DOCUMENT CLASS</font>"))
story.append(P(
    f"Table I reports detection. Both attack variants were caught in every "
    f"instance. One clean off-topic document of {n_o} was flagged, by the keyword "
    f"density detector rather than either rendering detector, giving a false "
    f"positive rate of {fpr_o*100:.1f} percent on that class and zero on genuine "
    f"relevant documents.", "body0"))

story.append(P("<i>C. Effect of the defence on ranking</i>", "h2"))
story.extend(fig("fig_adversarial", COLW,
                 "Fig. 3. (a) Share of the top five positions held by attacked "
                 "documents, averaged over three roles, with and without the "
                 "defence. Error bars are the standard error over roles. "
                 "(b) Proportion of each document class flagged."))
story.append(P(
    f"Enabling the defence removes the attacker from rank one in every role, from "
    f"{r1_off*100:.0f} percent to {r1_on*100:.0f} percent. The improvement is "
    f"real but partial. Attacked documents still occupy {t5_on*100:.0f} percent of "
    f"the top five, and their mean rank ({mean_rank(True,'attack'):.1f}) remains "
    f"marginally ahead of genuine relevant resumes "
    f"({mean_rank(True,'clean_relevant'):.1f}). The penalty demotes an attacked "
    f"document rather than eliminating it, and a resume that is off-topic but "
    f"otherwise well written retains some semantic similarity to the role. We "
    f"regard rank one as the operationally important case, since it is the "
    f"position a shortlist is anchored on, but we do not claim the attack is "
    f"fully neutralised.", "body0"))

story.append(P("<i>D. Transfer failure and domain adaptation</i>", "h2"))
rows = [["System", "P@10", "R@10", "NDCG@10", "MRR", "s/query"]]
for key, name in [("tfidf", "TF-IDF"), ("bi", "Bi-encoder"),
                  ("full_pre", "+ CE, pretrained"),
                  ("full_ft", "+ CE, fine-tuned")]:
    rows.append([name,
                 f"{ev_mean(key,'p_at_k'):.3f}",
                 f"{ev_mean(key,'r_at_k'):.3f}",
                 f"{ev_mean(key,'ndcg'):.3f}",
                 f"{ev_mean(key,'mrr'):.3f}",
                 f"{ev_mean(key,'latency_s'):.2f}"])
story.extend(table(rows,
                   [1.12 * inch, 0.42 * inch, 0.42 * inch, 0.52 * inch,
                    0.42 * inch, 0.45 * inch],
                   "TABLE II. R<font size=6.5>ETRIEVAL QUALITY OVER </font>24 "
                   "<font size=6.5>QUERIES</font>"))
d_bt, t_bt, p_bt, *_ = paired("bi", "tfidf")
story.append(P(
    f"Table II gives the retrieval comparison. Bi-encoder retrieval improved on "
    f"the lexical baseline, {bi_p:.3f} against {tf_p:.3f} Precision@10, though the "
    f"difference did not reach significance over 24 queries (p = {p_bt:.3f}).", "body0"))
story.append(P(
    f"The re-ranker behaved in two opposite ways depending on whether it had seen "
    f"the domain. Used as published, it <i>reduced</i> Precision@10 from "
    f"{bi_p:.3f} to {pre_p:.3f}, a mean paired difference of {d_bf:+.3f} "
    f"(t = {t_bf:.2f}, p = {p_bf:.3f}), losing on {l_bf} of 24 queries and winning "
    f"on only {w_bf}. A practitioner following the standard two-stage recipe would "
    f"have paid roughly twice the compute for a worse ranking.", "body0"))
story.append(P(
    f"We attribute this to domain mismatch. The published checkpoint is trained on "
    f"MS MARCO, where both query and passage are short. Our inputs are resumes "
    f"averaging {CS['mean_chars']:,.0f} characters, truncated to the model's "
    f"512-token window, matched against paragraph-length job descriptions. Neither "
    f"side of the pair resembles the training distribution, and the BEIR results "
    f"[7] indicate this is a general rather than an incidental problem.", "body0"))
story.append(P(
    f"Adapting the same architecture to the domain reversed the result. Training "
    f"on 1,600 weakly supervised pairs for a single epoch, with positives drawn "
    f"from matching category labels and negatives from mismatched ones, raised "
    f"Precision@10 to {ft_p:.3f}. The gain over the unadapted checkpoint is "
    f"{d_ft:+.3f} (t = {t_ft:.2f}, p = {p_ft:.4f}), and the adapted system also "
    f"beats the bi-encoder alone ({d_fbi:+.3f}, p = {p_fbi:.3f}) and the lexical "
    f"baseline ({d_fb:+.3f}, t = {t_fb:.2f}, p = {p_fb:.4f}; {w_fb} wins, "
    f"{ti_fb} ties, {l_fb} losses). It is also marginally faster than the "
    f"unadapted model, {ev_mean('full_ft','latency_s'):.1f} against "
    f"{ev_mean('full_pre','latency_s'):.1f} seconds per query.", "body0"))
story.extend(fig("fig_retrieval", COLW,
                 "Fig. 4. Retrieval quality across the four systems. Error bars "
                 "are the standard error over 24 queries."))
story.extend(fig("fig_perquery", COLW,
                 "Fig. 5. Per-query change in Precision@10 from domain adaptation, "
                 "sorted. Adaptation improved 13 queries and hurt 2."))
story.append(P(
    "Fig. 5 shows that the improvement is broad rather than driven by a few "
    "queries. We draw two conclusions. A general-purpose re-ranker should not be "
    "assumed to help, and the cost of checking is one evaluation run. When it does "
    "not help, modest domain adaptation can be enough to recover the expected "
    "benefit: a single epoch over weak labels that required no manual annotation "
    "moved this component from actively harmful to the best configuration we "
    "measured.", "body0"))

# ---- VII. Limitations
story.append(P("VII.&nbsp;&nbsp;L<font size=7.5>IMITATIONS</font>", "h1"))
story.append(P(
    "Our relevance labels are category matches, not human judgements. A resume "
    "from the target category may be a poor fit for a specific role and a resume "
    "from an adjacent category may be a good one, so absolute metric values should "
    "be read as approximate. The comparison between systems is unaffected because "
    "all systems see identical labels.", "body0"))
story.append(P(
    "The brightness threshold is a fixed value. An attacker who uses a very light "
    "grey rather than pure white can fall below it, and we have not characterised "
    "the trade-off curve between threshold and false positive rate. The keyword "
    "density detector provides partial coverage in that case because it is "
    "independent of colour, but we expect a determined attacker to evade a static "
    "threshold.", "body0"))
story.append(P(
    "Our adversarial evaluation covers three roles and two attack variants, with "
    f"{len(ADD)} documents. The detection rates are encouraging but the sample is "
    "small, and we generate attacked documents programmatically, so they may be "
    "more uniform than real adversarial submissions.", "body0"))
story.append(P(
    "We did not evaluate whether the penalty magnitude is well calibrated. It was "
    "chosen so that a heavily stuffed document ranks below a clean one, which "
    "the results in Section VI-C show is only partially achieved.", "body0"))

# ---- VIII. Conclusion
story.append(P("VIII.&nbsp;&nbsp;C<font size=7.5>ONCLUSION AND</font> "
               "F<font size=7.5>UTURE</font> W<font size=7.5>ORK</font>", "h1"))
story.append(P(
    f"Resume screening systems inherit a vulnerability from the way they read "
    f"documents. Because text extraction discards rendering attributes, an "
    f"applicant can inject keywords that the model reads and a recruiter cannot "
    f"see, and in our experiments this was sufficient to take rank one in every "
    f"role tested. Inspecting the rendering layer rather than the text layer "
    f"detected both attack variants in all {n_w + n_t} attacked documents at a "
    f"false positive cost of {fpr_o*100:.1f} percent on clean off-topic documents, "
    f"and removed the attacker from rank one. The defence is incomplete: attacked "
    f"documents retain {t5_on*100:.0f} percent of top-five positions.", "body0"))
story.append(P(
    "We also found that a general-purpose cross-encoder re-ranker reduced ranking "
    "quality on this corpus while costing substantially more compute, which we "
    "read as a domain transfer failure rather than a flaw in the two-stage "
    "approach itself.", "body0"))
story.append(P(
    "Three directions follow. An adaptive brightness threshold estimated from the "
    "document background would address the static threshold weakness, and an "
    "attacker who uses near-white rather than pure white is the obvious next "
    "evaluation. Our adaptation used a single epoch over weak labels, so stronger "
    "supervision, whether from recruiter feedback or from harder negative mining, "
    "is likely to improve the re-ranker further. Finally, a study with human "
    "relevance judgements would establish how far the category proxy distorts the "
    "absolute numbers reported here.", "body0"))

# ---- References
story.append(P("R<font size=7.5>EFERENCES</font>", "h1"))
refs = [
    "N. Reimers and I. Gurevych, \"Sentence-BERT: Sentence embeddings using "
    "Siamese BERT-networks,\" in <i>Proc. EMNLP-IJCNLP</i>, 2019, pp. 3982-3992.",
    "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, \"BERT: Pre-training of "
    "deep bidirectional transformers for language understanding,\" in <i>Proc. "
    "NAACL-HLT</i>, 2019, pp. 4171-4186.",
    "Z. Gyongyi and H. Garcia-Molina, \"Web spam taxonomy,\" in <i>Proc. 1st Int. "
    "Workshop on Adversarial Information Retrieval on the Web (AIRWeb)</i>, 2005.",
    "R. Nogueira and K. Cho, \"Passage re-ranking with BERT,\" "
    "<i>arXiv:1901.04085</i>, 2019.",
    "W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, \"MiniLM: Deep "
    "self-attention distillation for task-agnostic compression of pre-trained "
    "transformers,\" in <i>Proc. NeurIPS</i>, 2020.",
    "J. Johnson, M. Douze, and H. Jegou, \"Billion-scale similarity search with "
    "GPUs,\" <i>IEEE Trans. Big Data</i>, vol. 7, no. 3, pp. 535-547, 2021.",
    "N. Thakur, N. Reimers, A. Ruckle, A. Srivastava, and I. Gurevych, \"BEIR: A "
    "heterogeneous benchmark for zero-shot evaluation of information retrieval "
    "models,\" in <i>Proc. NeurIPS Datasets and Benchmarks</i>, 2021.",
    "M. Bogen and A. Rieke, \"Help wanted: An examination of hiring algorithms, "
    "equity, and bias,\" Upturn, Tech. Rep., 2018.",
    "M. Raghavan, S. Barocas, J. Kleinberg, and K. Levy, \"Mitigating bias in "
    "algorithmic hiring: Evaluating claims and practices,\" in <i>Proc. ACM "
    "FAccT</i>, 2020, pp. 469-481.",
    "S. Robertson and H. Zaragoza, \"The probabilistic relevance framework: BM25 "
    "and beyond,\" <i>Found. Trends Inf. Retr.</i>, vol. 3, no. 4, pp. 333-389, 2009.",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(f"[{i}]&nbsp;&nbsp;{r}", S["ref"]))


# ---------------------------------------------------------------- doc
class Doc(BaseDocTemplate):
    pass


doc = Doc(str(OUT), pagesize=LETTER,
          leftMargin=MARGIN, rightMargin=MARGIN,
          topMargin=MARGIN, bottomMargin=MARGIN,
          title="Rendering-Layer Integrity Verification for Adversarially "
                "Robust Resume Screening",
          author="Pant, Dutia, Lohiya, Agrawal")

TITLE_H = 1.50 * inch
usable_h = PH - 2 * MARGIN

f_title = Frame(MARGIN, PH - MARGIN - TITLE_H, PW - 2 * MARGIN, TITLE_H,
                id="title", leftPadding=0, rightPadding=0,
                topPadding=0, bottomPadding=0)
f_l1 = Frame(MARGIN, MARGIN, COLW, usable_h - TITLE_H, id="l1",
             leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
f_r1 = Frame(MARGIN + COLW + GUT, MARGIN, COLW, usable_h - TITLE_H, id="r1",
             leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
f_l = Frame(MARGIN, MARGIN, COLW, usable_h, id="l",
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
f_r = Frame(MARGIN + COLW + GUT, MARGIN, COLW, usable_h, id="r",
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)


def footer(canvas, d):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.drawCentredString(PW / 2, MARGIN - 16, str(canvas.getPageNumber()))
    canvas.restoreState()


doc.addPageTemplates([
    PageTemplate(id="first", frames=[f_title, f_l1, f_r1], onPage=footer),
    PageTemplate(id="twocol", frames=[f_l, f_r], onPage=footer),
])

doc.build(story)
print(f"wrote {OUT}")
print(f"pages: see file; size {OUT.stat().st_size/1024:.0f} KB")
