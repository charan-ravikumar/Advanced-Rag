"""The Evaluation tab. It reads ONLY the committed artifact (``app_data/results.json``, built from saved run folders by ``scripts/build_app_artifacts.py``) and never runs anything:
no retrieval, no model call, no network. Every table carries a scope caption (TEST / DEV / EXPLORATORY, split, setting, n, run folder, fingerprint); every number is formatted from the
artifact and from nothing else. A missing or partly invalid artifact shows a notice and whatever sections are valid.
"""

from __future__ import annotations

from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from app.artifacts import fmt_ms, scope_caption

METRIC_COLUMNS = [("nDCG@10", "nDCG@10"), ("MRR", "MRR"), ("HitRate@5", "HitRate@5"), ("EvidenceRecall@10", "EvidenceRecall@10"), ("EvidenceF1@5", "EvidenceF1@5")]
LABEL_HELP = {"TEST": "final split, run once", "DEV": "development split: configurations were chosen on it, so it is optimistic", "EXPLORATORY": "small sample, not a result"}

NOT_SHOWN = (
    "**No generation result on the TEST split.** The test split was run once, for retrieval only. The generation numbers below are on the dev split and are exploratory.",
    "**No baseline and no noise floor for generation.** One arm (the system as built), one seeded sample, one run: there is no oracle-context run, no title-and-abstract run and no repeat to "
    "show how much two identical runs differ. A difference between two numbers here would not be interpretable.",
    "**RAGAS scores are incomplete and uncalibrated.** The judge is an LLM; its scores have not been compared with human labels, so they are the judge's opinion, not a measurement of correctness. "
    "They are not shown unless a scoring run completed.",
    "**Answer-F1 is word overlap.** A correct answer in other words scores low; a long answer that happens to contain the reference words scores high. A valid citation shows which passage was "
    "pointed at, not that it supports the claim.",
    "**Retrieval metrics are about finding evidence paragraphs, not about answer quality.** A better retrieval number does not by itself mean better answers.",
    "**One dataset.** Qasper: English NLP papers, questions written from the title and abstract. Text evidence only (tables and figures are excluded). Results do not transfer to other domains or languages.",
    "**Latency is measured on one machine's CPU, retrieval only** (not end-to-end, not under load). The language-model call adds its own time on a free tier.",
    "**No comparison with other systems and no user study.**",
)


def _fmt(entry: dict | None, with_ci: bool = True) -> str:
    if not entry or entry.get("mean") is None:
        return "—"
    if with_ci and entry.get("ci_low") is not None:
        return f"{entry['mean']:.3f} [{entry['ci_low']:.3f}, {entry['ci_high']:.3f}]"
    return f"{entry['mean']:.3f}"


def _caption(section: dict) -> None:
    scope = section["scope"]
    st.caption(f":blue-background[{scope['label']}] {LABEL_HELP.get(scope['label'], '')}")
    st.caption(scope_caption(scope))
    if scope.get("note"):
        st.caption(scope["note"])


def arms_dataframe(arms: list[dict], with_ci: bool) -> pd.DataFrame:
    rows = []
    for a in arms:
        row = {"Arm": a["arm"], "Configuration": a["description"]}
        for key, label in METRIC_COLUMNS:
            row[label] = _fmt(a["metrics"].get(key), with_ci)
        lat = a.get("latency") or {}
        row["Retrieval latency p50 / p95"] = f"{fmt_ms(lat['total_p50_ms'])} / {fmt_ms(lat['total_p95_ms'])}" if lat else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def comparisons_dataframe(comparisons: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"Comparison": c["pair"], "Metric": c["metric"], "Mean first": f"{c['mean_x']:.3f}", "Mean second": f"{c['mean_y']:.3f}",
                          "Difference (first − second)": f"{c['diff']:+.3f}", "95% CI (paper-clustered)": f"[{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]",
                          "Holm-significant": "yes" if c["holm_significant"] else "no", "Wins / losses / ties": f"{c['wins']} / {c['losses']} / {c['ties']}",
                          "Questions / papers": f"{c['n']} / {c['n_papers']}"} for c in comparisons])


def _interval_chart(df: pd.DataFrame, category: str, mean: str, lo: str, hi: str, title: str, zero_line: bool = False):
    base = alt.Chart(df)
    bars = base.mark_rule().encode(y=alt.Y(f"{category}:N", sort=None, title=None), x=alt.X(f"{lo}:Q", title=title), x2=f"{hi}:Q")
    points = base.mark_point(filled=True, size=80).encode(y=alt.Y(f"{category}:N", sort=None), x=f"{mean}:Q", tooltip=[category, mean, lo, hi])
    layers = bars + points
    if zero_line:
        layers = layers + alt.Chart(pd.DataFrame({"zero": [0.0]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="zero:Q")
    return layers.properties(height=max(120, 46 * len(df)))


def _retrieval_test(section: dict) -> None:
    st.markdown("#### Retrieval on the TEST split")
    _caption(section)
    st.dataframe(arms_dataframe(section["arms"], with_ci=True), hide_index=True, width="stretch")
    st.caption("Mean per arm with a 95% paper-clustered bootstrap interval (papers are resampled, a paper's questions stay together).")
    df = pd.DataFrame([{"Arm": a["arm"], "mean": a["metrics"]["nDCG@10"]["mean"], "low": a["metrics"]["nDCG@10"]["ci_low"], "high": a["metrics"]["nDCG@10"]["ci_high"]}
                       for a in section["arms"]])
    st.altair_chart(_interval_chart(df, "Arm", "mean", "low", "high", "nDCG@10 with 95% interval"), width="stretch")
    st.markdown("##### Pre-registered comparisons")
    method = section["comparison_method"]
    st.caption(f"Difference = first arm minus second arm; {method['interval']} ({method['resamples']} resamples, seed {method['seed']}); Holm correction over "
               f"{method['holm_family_size']} comparisons at alpha {method['alpha']}. The comparisons were fixed before the test split was run.")
    st.dataframe(comparisons_dataframe(section["comparisons"]), hide_index=True, width="stretch")
    cdf = pd.DataFrame([{"Comparison": c["pair"], "mean": c["diff"], "low": c["ci_low"], "high": c["ci_high"]} for c in section["comparisons"]])
    st.altair_chart(_interval_chart(cdf, "Comparison", "mean", "low", "high", "Difference in nDCG@10 (first − second), 95% interval", zero_line=True), width="stretch")
    st.markdown("##### Retrieval latency (retrieval only, this machine's CPU)")
    lat_rows = []
    for a in section["arms"]:
        lat = a["latency"]
        lat_rows.append({"Arm": a["arm"], "Total p50": fmt_ms(lat["total_p50_ms"]), "Total p95": fmt_ms(lat["total_p95_ms"]),
                         "Rerank stage p50": fmt_ms(lat["rerank_p50_ms"]) if "rerank_p50_ms" in lat else "—",
                         "Rerank stage p95": fmt_ms(lat["rerank_p95_ms"]) if "rerank_p95_ms" in lat else "—", "Queries measured": lat.get("n_measured") or "—"})
    st.dataframe(pd.DataFrame(lat_rows), hide_index=True, width="stretch")


def _retrieval_dev(section: dict) -> None:
    st.markdown("#### Development ablation (all configurations tried)")
    _caption(section)
    st.dataframe(arms_dataframe(section["arms"], with_ci=False), hide_index=True, width="stretch")


def _setting_comparison(section: dict) -> None:
    st.markdown("#### One paper given vs all papers searched")
    _caption(section)
    rows = [{"Setting": e["setting"], "Arm": e["arm"], **{k: f"{v['mean']:.3f}" for k, v in e["metrics"].items()}} for e in section["entries"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("DocHit@5 is the share of questions for which a passage of the right paper was among the top 5 results (it is 1 by construction when the paper is given).")


def _generation(section: dict, ragas: dict | None) -> None:
    st.markdown("#### Generation (exploratory)")
    _caption(section)
    gen = section["generator"]
    st.caption(f"Generator: {gen['model']} at reasoning effort {gen['reasoning_effort']}, temperature {gen['temperature']}; {gen['top_k']} passages per question. {section['metrics_note']}")
    s, lng = section["short_mode"], section["long_mode"]
    rows = [{"Metric": "Answer-F1 (short answers)", "Value": _fmt(s["AnswerF1"]), "Questions": s["AnswerF1"]["n"]}]
    for typ, entry in s["AnswerF1_by_best_reference_type"].items():
        rows.append({"Metric": f"Answer-F1, best reference is {typ.replace('_', '/')}", "Value": _fmt(entry), "Questions": entry["n"]})
    rows += [{"Metric": "Yes/no accuracy", "Value": _fmt(s["YesNoAccuracy"]), "Questions": s["YesNoAccuracy"]["n"]},
             {"Metric": "Abstention rate on answerable questions (short)", "Value": _fmt(s["AbstentionRate_on_answerable"]), "Questions": s["AbstentionRate_on_answerable"]["n"]},
             {"Metric": "Abstention rate on answerable questions (long)", "Value": _fmt(lng["AbstentionRate_on_answerable"]), "Questions": lng["AbstentionRate_on_answerable"]["n"]}]
    pooled = lng["CitationValidity_pooled"]
    if pooled:
        rows.append({"Metric": "Citation validity (valid / cited labels, long answers)", "Value": f"{pooled['valid']} of {pooled['cited']}", "Questions": lng["CitationValidity_per_answer"]["n"]})
    none = lng["answers_with_no_citation"]
    rows.append({"Metric": "Long answers with no citation", "Value": f"{none['n']} of {none['of']}", "Questions": none["of"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    f1 = pd.DataFrame([{"Group": "all", **{k: s["AnswerF1"][k] for k in ("mean", "ci_low", "ci_high")}}] +
                      [{"Group": t, **{k: e[k] for k in ("mean", "ci_low", "ci_high")}} for t, e in s["AnswerF1_by_best_reference_type"].items() if e.get("ci_low") is not None])
    f1 = f1.rename(columns={"ci_low": "low", "ci_high": "high"})
    st.altair_chart(_interval_chart(f1, "Group", "mean", "low", "high", "Answer-F1 with 95% interval"), width="stretch")
    st.markdown("##### RAGAS (faithfulness, answer relevancy)")
    if not ragas:
        st.info("No RAGAS scoring run exists.")
    elif not ragas["complete"]:
        st.info(f"The RAGAS scoring is incomplete: {ragas['n_scored']} of {ragas['n_scorable']} answers were scored before the judge's quota ended"
                f"{' (' + ragas['stopped_reason'] + ')' if ragas.get('stopped_reason') else ''}, so no RAGAS number is shown. The judge ({ragas['judge']['model']}) is also uncalibrated. "
                f"Run folder: `{ragas['scope']['run_folder']}`.")
    else:
        st.caption(scope_caption(ragas["scope"]))
        m = ragas["metrics"]
        st.dataframe(pd.DataFrame([{"Metric": k, "Value": _fmt(v), "Answers": v["n"]} for k, v in m.items()]), hide_index=True, width="stretch")
        st.caption("UNCALIBRATED judge: these are the judge's scores, not a measurement of correctness.")


def _chunker(rows: list[dict]) -> None:
    st.markdown("#### Chunker findings")
    st.caption("Measurements of the chunking step, quoted from `docs/CLAIMS_LOG.md` (each row names its run folder and the code fingerprint).")
    st.dataframe(pd.DataFrame([{"Finding": r["metric"], "Value": r["value"], "Run folder": ", ".join(r["run_folders"]), "Fingerprint": r["code_fingerprint"][:12], "Scope": r["scope_note"]}
                               for r in rows]), hide_index=True, width="stretch")


def render_evaluation(res) -> None:
    arts = res.artifacts
    st.subheader("Evaluation")
    st.caption("This tab only displays saved results: it reads `app_data/results.json`, built from the run folders by `scripts/build_app_artifacts.py`. It never runs an evaluation.")
    for problem in arts.problems:
        st.warning(problem)
    results: dict[str, Any] | None = arts.results
    if not results:
        st.info("No evaluation results are available. Build them with `python scripts/build_app_artifacts.py`.")
    else:
        if results.get("retrieval_test"):
            _retrieval_test(results["retrieval_test"])
        if results.get("retrieval_dev"):
            _retrieval_dev(results["retrieval_dev"])
        if results.get("setting_comparison"):
            _setting_comparison(results["setting_comparison"])
        if results.get("generation"):
            _generation(results["generation"], results.get("ragas"))
        if results.get("chunker_findings"):
            _chunker(results["chunker_findings"])
    st.markdown("#### What the data does NOT show")
    for item in NOT_SHOWN:
        st.markdown(f"- {item}")
