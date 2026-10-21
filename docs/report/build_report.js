// Build the 4-page Word report for the moving-window extreme rainfall task.
//
// Usage (from the repository root, after scripts 01-05):
//     node docs/report/build_report.js
// Needs the npm package "docx" (npm install docx). Reads the report figures
// from outputs/figures/ and writes docs/report/<OUT_NAME>.
// Every number in the text comes from the tables in outputs/tables/.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, WidthType, ShadingType, BorderStyle,
  LevelFormat, Footer, PageNumber, ExternalHyperlink, TabStopType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = path.join(ROOT, "outputs", "figures");
const OUT_NAME = "Beikahmadi_moving-window_rainfall_extremes_report.docx";
const OUT = process.argv[2] || path.join(__dirname, OUT_NAME);
const REPO = "https://github.com/Niloufarbeikahmadi/ordinary-to-extreme";

// ---------------------------------------------------------------- geometry
const CM = 567;                                   // DXA per cm
const MARGIN = Math.round(1.8 * CM);
const TEXT_W = Math.round(21.0 * CM) - 2 * MARGIN; // A4 width minus margins
const FONT = "Calibri";
const INK = "1A1A1A", MUTED = "6B6A66", BAND = "F3F2EF", RULE = "C3C2B7";
const CAPTION = "3A3936";

// ------------------------------------------------- inline markup -> runs
// **bold**, *italic*, _{subscript}, ^{superscript}
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...base }));
    else if (tok.startsWith("*")) out.push(new TextRun({ text: tok.slice(1, -1), italics: true, ...base }));
    else if (tok.startsWith("_{")) out.push(new TextRun({ text: tok.slice(2, -1), subScript: true, ...base }));
    else out.push(new TextRun({ text: tok.slice(2, -1), superScript: true, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}),
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 70, before: opts.before ?? 0, line: 240 },
});
const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1, children: [new TextRun(text)],
  spacing: { before: 110, after: 50 }, keepNext: true,
});
const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, children: [new TextRun(text)],
  spacing: { before: 60, after: 30 }, keepNext: true,
});
const BULLET = (text, ref = "bullets") => new Paragraph({
  numbering: { reference: ref, level: 0 }, children: runs(text),
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 35, line: 240 },
});

function pngSize(buf) {                 // width, height from the IHDR chunk
  return [buf.readUInt32BE(16), buf.readUInt32BE(20)];
}

function figure(file, caption, widthCm = 17.4) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const [w, h] = pngSize(buf);
  const wpx = Math.round(widthCm / 2.54 * 96);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 60, after: 20 },
      keepNext: true,
      children: [new ImageRun({ type: "png", data: buf,
        transformation: { width: wpx, height: Math.round(wpx * h / w) },
        altText: { title: file, description: caption.replace(/\*/g, ""), name: file } })],
    }),
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 90, line: 228 },
      children: runs(caption, { size: 16, color: CAPTION }),
    }),
  ];
}

// ------------------------------------------------------------ equation
const eqBuf = fs.readFileSync(path.join(FIG, "report_equation.png"));
const [eqW0, eqH0] = pngSize(eqBuf);
const eqW = Math.round(6.4 / 2.54 * 96);
const equation = new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 20, after: 30 },
  children: [new ImageRun({ type: "png", data: eqBuf,
    transformation: { width: eqW, height: Math.round(eqW * eqH0 / eqW0) },
    altText: { title: "MEVD", name: "equation",
      description: "zeta(x) = (1/n) sum_j [1 - exp(-((x - theta)/C_j)^w_j)]^N_j" } })],
});

// --------------------------------------------------------------- table
const border = { style: BorderStyle.SINGLE, size: 4, color: RULE };
const noB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const COLS = [Math.round(TEXT_W * 0.44), Math.round(TEXT_W * 0.28)];
COLS.push(TEXT_W - COLS[0] - COLS[1]);
function cell(text, i, header = false, shade = false) {
  return new TableCell({
    width: { size: COLS[i], type: WidthType.DXA },
    borders: { top: noB, left: noB, right: noB, bottom: header ? border : noB },
    shading: shade ? { fill: BAND, type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 22, bottom: 22, left: 80, right: 80 },
    children: [new Paragraph({
      alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER, keepNext: true,
      children: runs(text, { size: 16, bold: header }),
    })],
  });
}
const tableRows = [
  ["", "Bologna", "Padova"],
  ["Record; valid years; retained windows", "1813–2007; 195; 166", "1725–2019; 281; 261"],
  ["GEV 200-yr level: range (median) [mm]", "72–316 (141)", "69–277 (124)"],
  ["MEVD 200-yr level: range (median) [mm]", "130–177 (152)", "106–176 (135)"],
  ["Median 90% band width / estimate: GEV | MEVD", "1.20 | 0.24", "1.09 | 0.15"],
  ["Last window: MEVD [90% band] (mm)", "1978–2007: 152 [142–160]", "1990–2019: 170 [153–185]"],
  ["Last window: GEV [90% band] (mm)", "129 [90–204]", "186 [106–398]"],
  ["50-yr level exceeded out of sample: GEV | MEVD (expected)", "7 | 3 (3.7)", "11 | 4 (5.3)"],
  ["Coverage of nominal 90% bands, synthetic: GEV | MEVD", "0.84 | 0.74", "0.85 | 0.68"],
  ["1948–77 → 1978–2007: MEVD | wet-day scale C", "+3% [−10, +19] | +15% [+8, +22]", "+21% [+3, +35] | +11% [+4, +19]"],
];
const table = new Table({
  width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: COLS,
  borders: { top: border, bottom: border, left: noB, right: noB,
             insideHorizontal: noB, insideVertical: noB },
  rows: tableRows.map((r, k) => new TableRow({
    cantSplit: true,
    children: r.map((t, i) => cell(t, i, k === 0, k > 0 && k % 2 === 0)),
  })),
});

// ------------------------------------------------------------- content
const children = [];
children.push(new Paragraph({
  spacing: { after: 60 }, alignment: AlignmentType.LEFT,
  children: [new TextRun({ text: "How rare is rare? The 200-year daily rainfall of Bologna and Padova in moving 30-year windows: GEV versus MEVD-Weibull", bold: true, size: 30, color: INK })],
}));
children.push(new Paragraph({
  spacing: { after: 20 },
  children: [new TextRun({ text: "Niloufar Beikahmadi  ·  Test task, Young Research Associate on sub-hourly extreme rainfall, CRITICAL, University of Padova  ·  October 2026", size: 17, color: MUTED })],
}));
children.push(new Paragraph({
  spacing: { after: 90 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 4 } },
  children: [
    new TextRun({ text: "Code, tables and all figures (reproducible from the two data files): ", size: 17, color: MUTED }),
    new ExternalHyperlink({ link: REPO, children: [new TextRun({ text: REPO.replace("https://", ""), size: 17, style: "Hyperlink" })] }),
  ],
}));

const keyItems = [
  "**Stability.** Over the same windows the GEV 200-year daily depth spans 72–316 mm (Bologna) and 69–277 mm (Padova), the MEVD 130–177 and 106–176 mm. One year can halve the GEV estimate: Bologna drops from 136 to 72 mm when 24 February 1933 leaves the window.",
  "**Uncertainty.** With identical resampled years, MEVD 90% bands are 5–7 times narrower than GEV bands. Out of sample, the GEV 50-year level is exceeded about twice as often as expected (Padova 11 vs 5.3, p = 0.02), the MEVD level is not. Both bands are too narrow in synthetic tests (GEV 80–85%, MEVD 68–74% coverage), and the MEVD’s modelling choices move it more than sampling noise does.",
  "**Two stations.** Wet-day occurrence co-varies (r = 0.61), annual maxima barely (0.18): most changes of the 200-year level are station-specific. One change is shared: from 1948–1977 to 1978–2007 wet days became more intense at both stations (Weibull scale C +15% and +11%, 90% intervals excluding zero).",
  "**Padova now.** The latest windows are Padova’s highest (MEVD 170 mm [153–185] for 1990–2019), carried by heavier, not more frequent, wet days and not by the single 2009 storm.",
];
children.push(new Table({
  width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W],
  borders: { top: noB, bottom: noB, left: noB, right: noB, insideHorizontal: noB, insideVertical: noB },
  rows: [new TableRow({ children: [new TableCell({
    width: { size: TEXT_W, type: WidthType.DXA },
    shading: { fill: BAND, type: ShadingType.CLEAR, color: "auto" },
    borders: { top: noB, bottom: noB, right: noB, left: { style: BorderStyle.SINGLE, size: 18, color: "EB6834" } },
    margins: { top: 70, bottom: 50, left: 140, right: 140 },
    children: [new Paragraph({ spacing: { after: 30 }, children: [new TextRun({ text: "Key results", bold: true, size: 19 })] }),
               ...keyItems.map((t) => BULLET(t, "keys"))],
  })] })],
}));

children.push(H1("1  Introduction"));
children.push(P("Design rainfall is usually extrapolated from a few decades of annual maxima with the Generalized Extreme Value (GEV) distribution, an asymptotic model that assumes each year holds very many independent events. The Metastatistical Extreme Value Distribution (MEVD; Marani & Ignaccolo, 2015; Zorzetto et al., 2016) builds the distribution of annual maxima from all ordinary wet days, year by year, without that assumption. We compare the two in 30-year moving windows over the Bologna and Padova daily records: how the 200-year daily depth and its uncertainty change through time, and whether the changes are shared by the two stations. Daily data are a proving ground for short sub-hourly records and for the evaluation of convection-permitting models (CPMs), where 10–30-year samples are the norm."));

children.push(H1("2  Data and quality control"));
children.push(P("**Bologna**: GHCN-Daily ITE00100550, 1813–2007 (tenths of mm, divided by 10), 195 complete years. **Padova**: the series of Marani & Zanetti (2015), 1725–2019, 281 valid years; 14 years are missing (1765–67, 1815–22, 1838–40). A year is valid with less than 10% of days missing (Zorzetto et al., 2016); since data are missing in whole years, 5% or 20% select the same years. Windows of 30 calendar years, advanced by one year, were kept with at least 20 valid years (Bologna 166 of 166, Padova 261 of 266) and are plotted at their final year. Quality control flagged, without deleting anything: in Padova, 1823–1837 and 1841 with half the usual wet days (43 vs 80 a year) but heavier ones (17.6 vs 10.3 mm) and near-normal totals, the signature of multi-day amounts read on one day (della Valle et al., 2023); a 31-day block of 1748 repeated in 1749; eight recording resolutions, including exact multiples of 1/1000 English inch in 1725–1764; and too few small rains in 1844–1864, when the observatory roof served as the funnel. In Bologna, 0.5–1 mm resolution before about 1880 raises the effective wet-day threshold to ~1.5 mm, and three annual maxima (1830, 1860, 1929) are isolated winter spikes after at least 7 dry days, likely snow measured on one day. The main analysis uses the record as provided, as Marani & Zanetti (2015) did; a QC-screened record is a sensitivity case."));

children.push(H1("3  Methods"));
children.push(P("**GEV.** Fitted by maximum likelihood (ML) to the annual maxima of each window, as in Marani & Zanetti (2015); L-moments as sensitivity. With their conventions our Padova 200-year levels span 68.7–276.7 mm, against their ~70–280 mm. **MEVD-Weibull.** In valid year *j* the N_{j} days with P > 1 mm are the ordinary events; a Weibull with scale C_{j} and shape w_{j} is fitted to the excesses P − 1 mm by probability weighted moments (PWM), as in mevpy. For the *n* valid years of a window,", { after: 20 }));
children.push(equation);
children.push(P("and the 200-year level solves ζ(x) = 1 − 1/200, with θ = 1 mm added back. We re-implemented the mevpy formulation in Python and adapted it to moving windows; it reproduces mevpy to 6 × 10^{−11} mm in all 427 windows. As sensitivity, the Weibull was also fitted to the wet days of 3, 5 or 30 pooled years (estimation window EW; Miniussi & Marani, 2020) and in the simplified MEV (Marra et al., 2019). **Common bootstrap.** In each window the *n* valid years were resampled with replacement 500 times; each drawn year carries its annual maximum (GEV) and its (N, C, w) (MEVD), so both models see identical synthetic samples, and the pointwise 90% interval is the 5th–95th percentile of the 500 estimates. The 213,500 GEV refits use a vectorised Levenberg–Marquardt ML fitter with analytic derivatives and shape restricted to [−1, 1], where the likelihood is bounded (Smith, 1985); it matches the reference fitter to 5 × 10^{−5} mm. **Checks.** (i) Leave-one-year-out (LOYO): each annual maximum is compared with models fitted to the other years of its 30-year window. (ii) Coverage: 500 synthetic 30-year records per case, from a GEV world and from an ordinary-day world built from each station’s yearly (N, C, w), analysed with the same code. (iii) Sensitivity: one analysis choice changed at a time. (iv) Change: two disjoint 30-year windows are bootstrapped independently and the 90% interval of their difference is reported."));

children.push(H1("4  Results"));
children.push(H2("4.1  Bologna"));
children.push(P("The GEV level ranges from 72 to 316 mm (median 141 mm; Fig. 1a) and moves in steps as single years enter or leave the window. The record day of 24 February 1933 (155.7 mm) dominates: window 1933–1962 gives 136 mm; the next, without 1933, gives 72 mm with shape ξ = −0.51, a bounded tail whose upper limit (74 mm) lies below a value observed one year earlier. Its 90% band, 67–77 mm, is falsely precise, because resampling cannot create a value the window does not contain. The MEVD stays within 130–177 mm (median 152 mm). It is highest in windows ending 1900–1961, when the yearly Weibull tails are heaviest (mean w 0.82 against 0.86 in the other windows). The last window (1978–2007) gives 152 mm [142–160] for the MEVD and 129 mm [90–204] for the GEV."));

children.push(...figure("report_fig1_bands.png",
  "**Figure 1.** 200-year daily rainfall in 30-year moving windows, plotted at the final year, for (a) Bologna and (b) Padova: GEV fitted by ML to the annual maxima (blue) and MEVD-Weibull (orange), with pointwise 90% bands from 500 resamplings of the same years for both models. Triangles: GEV 95th percentile above 360 mm. Dots: annual maxima at their own year (diamonds: flagged and retained; crosses: Padova 1823–1837). Grey shading: Padova observer eras. Gaps: windows with fewer than 20 valid years."));

children.push(H2("4.2  Padova"));
children.push(P("The GEV ranges from 69 to 277 mm (median 124 mm; Fig. 1b); its maximum (1947–1976, ξ = +0.48) has a 90% band of 118–927 mm. The MEVD ranges from 106 to 176 mm (median 135 mm). Its minimum (106 mm, 1838–1867) falls in the windows holding the roof-funnel years, when small rains were under-recorded (1844–1864) and the yearly Weibull tail looks lighter (median w 1.0). From the 1960s the MEVD rises, to 170–176 mm in the windows ending 2006–2019; the last window (1990–2019) gives 170 mm [153–185] (GEV 186 mm [106–398]). The rise comes from the Weibull scale (mean C 9.3 mm in 1990–2019 against 7.9 mm in 1960–1989), not from the number of wet days (80 vs 81 per year), and it does not hinge on the 2009 storm: with 0 or 1 copies of 2009 in the resamples the MEVD median is 167 or 169 mm, the GEV 123 or 186 mm. Against the disjoint window 1960–1989 the MEVD rises by 14% (90% interval −2 to +29%; one Weibull per window: +14% [+2, +24%]) and C by 18% [+11, +26%]."));

children.push(new Paragraph({ spacing: { before: 40, after: 30 }, keepNext: true,
  children: runs("**Table 1.** Main numbers (200-year daily rainfall; bands and intervals are bootstrap 5th–95th percentiles).", { size: 16, color: CAPTION }) }));
children.push(table);
children.push(new Paragraph({ spacing: { after: 20 }, children: [] }));

children.push(H2("4.3  Uncertainty, calibration and sensitivity"));
children.push(P("The median 90% band is 120% (Bologna) and 109% (Padova) of the GEV estimate, against 24% and 15% for the MEVD. Because both models use the same resamples, the band of their difference is also available: it excludes zero in 18 of 166 Bologna and 32 of 261 Padova windows, mostly where the GEV tail is bounded and below the MEVD. The GEV band is a mixture: in Bologna 1933–1962 the 1933 maximum is missing from 39% of the resamples, and the GEV median is 72, 139, 178 and 252 mm with 0, 1, 2 and ≥3 copies of it (MEVD 132 to 188 mm; Fig. 2a). Out of sample (Fig. 2b), the GEV 50-year level was exceeded 7 times in 184 Bologna years (3.7 expected; p = 0.10) and 11 times in 263 Padova years (5.3; p = 0.02), the MEVD level 3 and 4 times. In synthetic records with a known truth (Fig. 2c), nominal 90% GEV bands contained it in 80–85% of cases, missing mostly on the high side, as Kyselý (2008, 2010) found; MEVD bands in 68–74%, missing on the low side because the yearly-fit MEVD is 13% too high, although its error stays half the GEV’s (relative RMSE 21–24% vs 36–52%). The bias is estimation noise: with ~70 wet days a year the yearly w is noisy (75% of its year-to-year variance in Bologna, 41% in Padova), and spurious heavy-tailed years dominate the MEVD tail; three years carry 75% of the 200-year exceedance probability of Bologna 1948–1977. With identical synthetic years, yearly fits overestimate the 200-year level by 16–19%; one Weibull for all years does not. On the real records, however, pooling wet days over 3, 5 or 30 years lowers the MEVD by 12–35 mm and lets Bologna’s 50-year level be exceeded 9–11 times instead of 3.7 (p ≤ 0.012), while Padova is unchanged (4 vs 5.3). Bologna’s largest wet days are heavier-tailed than the bulk (Weibull shape of the top 25%: 0.77 against 0.82, in every window), so the noise of yearly fits partly compensates a real tail that one pooled Weibull misses. The yearly fit is kept, in line with Miniussi & Marani (2020)."));
children.push(P("**Sensitivity** (one choice at a time). For the GEV, sampling dominates (median half-width of the 90% band: 86 mm in Bologna, 68 mm in Padova): the estimator, the QC screening and the window definition change the median window by about 1 mm (L-moments up to 77 and 94 mm in single windows), while 20-year windows raise the coefficient of variation across Bologna windows from 0.32 to 0.53 (50 years: 0.18). For the MEVD, modelling choices move the estimate as much as or more than sampling noise (half-width 18 and 10 mm): raw values instead of excesses (−29 and −24 mm, Bologna and Padova), ML instead of PWM (−26 and −11 mm), the estimation window (−12 to −35 mm), a fit to the top 25% of wet days (−22 and −24 mm) and a 1 mm recording resolution (−13 mm at both stations). QC screening, window definition and window length (at most 3.3 mm in the median window) matter little, and the valid-year rule not at all. Reading Padova 1725–1764 in Paris instead of English inches raises the levels of windows within that period by 6.6% (GEV) and 8% (MEVD)."));

children.push(...figure("report_fig2_uncertainty.png",
  "**Figure 2.** (a) Bologna 1933–1962: the 500 bootstrap 200-year levels grouped by how many copies of 1933 were drawn (GEV blue, MEVD orange; share of resamples at right). (b) Leave-one-year-out: observed/expected number of years exceeding the 10- and 50-year levels fitted on the other years of their window, for the GEV, the MEVD and the MEVD with one Weibull per window (circled: binomial p < 0.05). (c) Coverage of nominal 90% bands in 500 synthetic 30-year records per case (bars: ±1.96 Monte Carlo standard errors)."));

children.push(H1("5  Bologna–Padova comparison and interpretation"));
children.push(P("**Magnitude.** Over the 163 common windows (final years 1845–2007) the MEVD is higher in Bologna in 82% of them (medians 152 vs 134 mm), but the two 90% bands overlap in 80% of them (GEV: 99%). **Shared or station-specific?** Wet-day occurrence is regional: the year-to-year rank correlation between the stations is 0.61 for N and 0.46 for annual totals (184 common years), and 0.70 for the 30-year means of N. Intensity and extremes are much less shared: 0.21 for the yearly C and w, 0.18 for annual maxima and 0.15 for the MEVD window series (GEV −0.14; window series hold only ~5 independent values). Daily extremes at stations ~110 km apart come from different storms, so most temporal changes of the 200-year level are station-specific. The nineteenth-century features (Bologna’s coarse readings, Padova’s 1823–1837 accumulated readings and 1844–1864 small-rain deficit) coincide with observational changes, and the coarsening test shows that a 1 mm resolution alone lowers the MEVD by 13 mm; they should not be read as climate before homogenisation."));
children.push(P("**A shared change.** Between the disjoint windows 1948–1977 and 1978–2007 the wet-day scale C increased at both stations (Bologna +15% [+8, +22%], Padova +11% [+4, +19%]) while wet days became fewer (−4.5% [−10, +1%] and −8% [−14, −2%]; Fig. 3). In Padova the higher C is already present in 1978–1996 (8.9 against 8.0 mm), before the station moved in 1997, and Bologna shows the same step (8.2 against 7.2 mm): a regional signal is more likely than an instrumental one. The MEVD turns it into +21% [+3, +35%] in Padova. In Bologna the yearly-fit MEVD barely moves (+3% [−10, +19%]) because three heavy-tailed years hold up its 1948–1977 level, whereas the pooled MEVD rises by 17% [+2, +30%]. The GEV cannot resolve any of these changes (90% intervals from −39% to +107% and wider). **Asymptotic versus non-asymptotic.** The GEV extrapolates from 20–30 maxima with a poorly constrained shape, so one year can halve or double the 200-year level, a negative shape produces bounded and falsely precise bands, and rare events are under-predicted out of sample. The MEVD uses ~2,000 wet days per window: estimates are stable, bands narrow, and changes can be attributed to occurrence (N) or intensity (C, w). The price is that its uncertainty moves from sampling to modelling choices (estimation window, a Weibull fitted to bulk or tail), which must be reported with it."));

children.push(...figure("report_fig3_comparison.png",
  "**Figure 3.** 30-year means of (a) the yearly Weibull scale C and (b) the number of wet days N (Bologna solid, Padova dashed; shading: Padova windows holding the 1823–1837 accumulated readings, off scale). (c) Change between disjoint 30-year windows with 90% bootstrap intervals, for the GEV and MEVD 200-year levels and for N and C (arrows: interval beyond ±60%)."));

children.push(H1("6  Limitations and next steps"));
children.push(P("Wet days within a year are treated as independent and identically distributed (storm clustering and seasonality are neglected), and years within a window as exchangeable. The nonparametric bootstrap under-covers (Section 4.3), pointwise bands of windows sharing 29 years are not independent tests, and the disjoint-window intervals inherit the same approximation. The early records are not homogenised: the reconstruction of Padova 1812–1864 by della Valle et al. (2023) was not used, and the unit of the 1725–1764 readings is uncertain. All results concern daily totals. **Next steps.** (1) A non-stationary MEVD with C (and N) as functions of time, to test the post-1970s change formally at both stations. (2) Parametric or profile-likelihood intervals for the GEV, and a two-stage bootstrap (years, then days) for the MEVD. (3) Homogenised inputs and a resolution-aware wet-day threshold. (4) Sub-hourly data: the same framework on ARPAV sub-hourly gauges in Veneto, with storm-based ordinary events and the simplified MEV for durations from minutes to hours (Marra et al., 2019, 2020). (5) CPM evaluation: compare the ordinary-event statistics (N, C, w) of convection-permitting simulations with the gauges, to find which part of the extremes a model gets wrong rather than only whether its 200-year level is off."));

children.push(H1("References"));
const refs = [
  "della Valle, A., Camuffo, D., Becherini, F., et al. (2023). Recovering, correcting, and reconstructing precipitation data affected by gaps and irregular readings: the Padua series from 1812 to 1864. *Climatic Change*, 176, 9.",
  "Kyselý, J. (2008). A cautionary note on the use of nonparametric bootstrap for estimating uncertainties in extreme-value models. *Journal of Applied Meteorology and Climatology*, 47, 3236–3251.",
  "Kyselý, J. (2010). Coverage probability of bootstrap confidence intervals in heavy-tailed frequency models, with application to precipitation data. *Theoretical and Applied Climatology*, 101, 345–361.",
  "Marani, M., & Ignaccolo, M. (2015). A metastatistical approach to rainfall extremes. *Advances in Water Resources*, 79, 121–126.",
  "Marani, M., & Zanetti, S. (2015). Long-term oscillations in rainfall extremes in a 268 year daily time series. *Water Resources Research*, 51, 639–647.",
  "Marra, F., Zoccatelli, D., Armon, M., & Morin, E. (2019). A simplified MEV formulation to model extremes emerging from multiple nonstationary underlying processes. *Advances in Water Resources*, 127, 280–290.",
  "Marra, F., Borga, M., & Morin, E. (2020). A unified framework for extreme subdaily precipitation frequency analyses based on ordinary events. *Geophysical Research Letters*, 47, e2020GL090209.",
  "Miniussi, A., & Marani, M. (2020). Estimation of daily rainfall extremes through the metastatistical extreme value distribution: uncertainty minimization and implications for trend detection. *Water Resources Research*, 56, e2019WR026535.",
  "Smith, R. L. (1985). Maximum likelihood estimation in a class of nonregular cases. *Biometrika*, 72, 67–90.",
  "Zorzetto, E., Botter, G., & Marani, M. (2016). On the emergence of rainfall extremes from ordinary events. *Geophysical Research Letters*, 43, 8076–8082. Python implementation: mevpy, github.com/EnricoZorzetto/mevpy.",
];
for (const r of refs) {
  children.push(new Paragraph({
    children: runs(r, { size: 14 }), alignment: AlignmentType.LEFT,
    indent: { left: 284, hanging: 284 }, spacing: { after: 15, line: 216 },
  }));
}

// -------------------------------------------------------------- document
const doc = new Document({
  creator: "Niloufar Beikahmadi",
  title: "Moving-window extreme rainfall: Bologna and Padova",
  description: "GEV vs MEVD-Weibull 200-year daily rainfall in 30-year moving windows",
  styles: {
    default: { document: { run: { font: FONT, size: 19, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: FONT, color: INK },
        paragraph: { spacing: { before: 110, after: 50 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 19, bold: true, italics: true, font: FONT, color: INK },
        paragraph: { spacing: { before: 60, after: 30 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "keys", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 284, hanging: 200 } } } }] },
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN, footer: 400 } } },
    footers: { default: new Footer({ children: [new Paragraph({
      tabStops: [{ type: TabStopType.RIGHT, position: TEXT_W }],
      children: [
        new TextRun({ text: "N. Beikahmadi · Moving-window rainfall extremes, Bologna and Padova", size: 15, color: MUTED }),
        new TextRun({ text: "\t", size: 15 }),
        new TextRun({ children: [PageNumber.CURRENT, " / ", PageNumber.TOTAL_PAGES], size: 15, color: MUTED }),
      ] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
