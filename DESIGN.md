---
name: Fab inspection monitor
description: A completed SECOM batch read as a wafer bin map; bin colours carry state, graphite carries ink.
colors:
  graphite: "#1E2328"
  bin-fail: "#C43C3C"
  bin-alarm: "#C9851F"
  bin-alarm-status: "#B8741A"
  bin-pass-light: "#B9D9C2"
  bin-pass-status: "#2F8F4E"
  steel: "#4A6A8A"
  silicon-ground: "#E4E7EA"
  silicon-panel: "#F4F6F7"
  hairline: "#CBD1D6"
  chart-grid: "#D9DDE1"
  muted-ink: "#5B656E"
  pass-point: "#8C969F"
  random-baseline: "#9AA3AB"
  chip-warn-bg: "#F3E2C4"
  chip-warn-ink: "#7A4A0C"
  chip-ok-bg: "#D5EBDC"
  chip-ok-ink: "#1F6136"
  rail-raised: "#2B3238"
  rail-rule: "#3A424A"
  rail-ink: "#E6EAED"
  rail-accent: "#A9D3B4"
typography:
  display:
    fontFamily: "Public Sans, sans-serif"
    fontSize: "2rem"
    fontWeight: 700
  headline:
    fontFamily: "Public Sans, sans-serif"
    fontSize: "1.375rem"
    fontWeight: 650
  title:
    fontFamily: "Public Sans, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 600
  lede:
    fontFamily: "Public Sans, sans-serif"
    fontSize: "1.06rem"
    fontWeight: 400
    lineHeight: 1.5
  body:
    fontFamily: "Public Sans, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.6
  readout-value:
    fontFamily: "Public Sans, sans-serif"
    fontSize: "1.45rem"
    fontWeight: 650
    letterSpacing: "-0.01em"
    fontFeature: "tnum"
  label:
    fontFamily: "Public Sans, sans-serif"
    fontSize: "0.82rem"
    fontWeight: 500
  chip:
    fontFamily: "Public Sans, sans-serif"
    fontSize: "0.72rem"
    fontWeight: 650
    letterSpacing: "0.04em"
  measure:
    fontFamily: "JetBrains Mono, monospace"
    fontSize: "0.86em"
    fontWeight: 400
rounded:
  cell: "0px"
  swatch: "1px"
  chip: "4px"
  panel: "6px"
spacing:
  chip-y: "2px"
  legend-row: "6px"
  chip-x: "8px"
  cell-y: "12px"
  cell-x: "16px"
  legend-col: "18px"
components:
  readout-table:
    backgroundColor: "{colors.silicon-panel}"
    textColor: "{colors.graphite}"
    rounded: "{rounded.panel}"
    padding: "12px 16px"
  readout-label:
    textColor: "{colors.muted-ink}"
    typography: "{typography.label}"
  readout-value:
    textColor: "{colors.graphite}"
    typography: "{typography.readout-value}"
  chip-warn:
    backgroundColor: "{colors.chip-warn-bg}"
    textColor: "{colors.chip-warn-ink}"
    typography: "{typography.chip}"
    rounded: "{rounded.chip}"
    padding: "2px 8px"
  chip-ok:
    backgroundColor: "{colors.chip-ok-bg}"
    textColor: "{colors.chip-ok-ink}"
    typography: "{typography.chip}"
    rounded: "{rounded.chip}"
    padding: "2px 8px"
  button:
    backgroundColor: "{colors.silicon-panel}"
    textColor: "{colors.graphite}"
    rounded: "{rounded.panel}"
  sidebar-rail:
    backgroundColor: "{colors.graphite}"
    textColor: "{colors.rail-ink}"
  die-cell:
    rounded: "{rounded.cell}"
---

# Design System: Fab inspection monitor

## Overview

**Creative North Star: "The Probe-Station Bin Map"**

The batch is read the way a yield engineer reads a lot: as a wafer bin map. Every unit is one square die cell laid on a serpentine probe path in time order inside a notched wafer outline, and state is carried by bin colour the way a sort map carries it. The world is polished-silicon grey with lighter panels, graphite ink and a graphite control rail, and hairline rules for structure. Nothing is decorative; colour appears only where it means a bin, an alarm or a drift state.

Density is engineer-grade: a readout spec table instead of KPI tiles, tabular figures, monospace for every measurement, unit ID and sensor name, and live counts in tab labels. The surface is flat. Depth comes from tonal layering (ground, panel, rail), never from drop shadows.

**Key Characteristics:**
- Wafer bin map as the primary reading of a batch: square cells, notched circular outline, shape plus colour legend with counts.
- Three-tone tonal stack: silicon ground, lighter panel, graphite rail.
- Bin colours (red fail, amber alarm/drift, pale green pass) are the only chromatic colour; graphite does all other work.
- Public Sans for interface text, JetBrains Mono only for measurements and identifiers.
- Flat surfaces, hairline borders, 6px panel corners, square data cells.

## Colors

A cool, low-chroma silicon palette where graphite carries ink and structure and a small set of bin colours carry state.

### Primary
- **Graphite** (`graphite`): ink, headings, links, the sidebar rail, Streamlit's primary colour, queued and false-alarm die cells, chart bars, limit and budget rules, the selected-die halo. The system's one structural colour.

### Secondary (bin state)
- **Bin Red** (`bin-fail`): recorded failures only: caught-failure cells (filled), missed-failure cells (outline), failure points in control charts, the FAIL text in the queue table.
- **Bin Amber** (`bin-alarm`): MSPC alarm and drift in charts: Pareto cumulative line, PSI bars, the MSPC line in the budget frontier.
- **Status Amber** (`bin-alarm-status`): the same amber role inside Streamlit status widgets (theme orange/yellow), a slightly deeper value for text legibility on the panel.
- **Pale Bin Green** (`bin-pass-light`): not-queued die cells and the text-selection highlight. Pass reads as quiet, not celebratory.
- **Status Green** (`bin-pass-status`): Streamlit success states only.

### Tertiary
- **Steel** (`steel`): the third policy series (hybrid blend) and Streamlit's blue slot. Never a bin.

### Neutral
- **Silicon Ground** (`silicon-ground`): page background, dataframe header band.
- **Silicon Panel** (`silicon-panel`): readout table, widgets, secondary background.
- **Hairline** (`hairline`): every border and table rule; chart domain and ticks.
- **Chart Grid** (`chart-grid`): Altair grid lines, one step lighter than the hairline.
- **Muted Ink** (`muted-ink`): labels, captions, byline, legend text, axis labels.
- **Pass Point** (`pass-point`): pass points on T² / SPE control charts.
- **Random Baseline** (`random-baseline`): the random-expected series in the frontier.
- **Chip tints** (`chip-warn-bg`/`chip-warn-ink`, `chip-ok-bg`/`chip-ok-ink`): the drift status chip.
- **Rail tones** (`rail-raised`, `rail-rule`, `rail-ink`, `rail-accent`): sidebar raised surface, rules, text, and the pale green control accent on graphite.

### Named Rules
**The Bin Colour Rule.** Chromatic colour means state. Red is a recorded failure, amber is an MSPC alarm or drift, pale green is pass or not queued. A colour that carries no bin meaning is graphite or a neutral.

**The Shape Plus Colour Rule.** No state is shown by colour alone: missed failures are outlined cells, control-chart failures are triangles against circles, policy lines differ by dash, legends carry counts.

## Typography

**Display Font:** Public Sans (with sans-serif)
**Body Font:** Public Sans (with sans-serif)
**Label/Mono Font:** JetBrains Mono (with monospace)

**Character:** A plain, civic grotesque for reading and operating, paired with a technical mono reserved for numbers and names that an engineer would copy into a log.

### Hierarchy
- **Display** (700, 2rem): the page title.
- **Headline** (650, 1.375rem): Streamlit h2.
- **Title** (600, 1.125rem): tab section questions and About headings (h3/h4).
- **Lede** (400, 1.06rem, 1.5): the one finding sentence under the title, bold figures at 650, max 68ch.
- **Body** (400, 15px base, 1.6 in prose): About prose at max 70ch; byline at 0.86rem muted, max 92ch.
- **Readout value** (650, 1.45rem, -0.01em, tabular figures): readout numbers, with a 0.9rem/500 muted denominator.
- **Label** (500, 0.82rem, muted): readout row labels, legend, notes (0.8rem).
- **Chip** (650, 0.72rem, 0.04em, uppercase text): status chips only.
- **Measure** (JetBrains Mono, 0.86em): unit IDs, legend counts, sensor names on chart axes.

### Named Rules
**The Mono Means Measured Rule.** JetBrains Mono appears only on measurements, unit IDs and sensor names. Interface text, headings and labels stay in Public Sans.

**The Tabular Figures Rule.** Any column or readout of numbers uses tabular numerals so values align across rows.

## Layout

Wide layout capped at 1480px with 2.2rem top padding. The first viewport is a title, the finding sentence and a byline, then a 58/42 split with large gap: the wafer bin map (square aspect, max 600px) with its legend and note on the left, the readout table, previous/next/unit picker and contribution bars on the right. Working views live in tabs below, each label carrying a live count (`Queue · 62`). Controls live only in the graphite sidebar. Readout cells pad 12px 16px; legend items gap 6px vertically and 18px horizontally. On narrow screens Streamlit stacks the columns and collapses the rail.

## Elevation & Depth

Flat. There are no drop shadows anywhere in the build. Depth is tonal: silicon ground under silicon panels, with the graphite rail as the darkest plane. Hairline 1px borders separate surfaces. The only inset box-shadow draws the outline swatch in the legend, which is a stroke, not elevation.

### Named Rules
**The Tonal Stack Rule.** Separate planes by tone and hairline, not by shadow: ground, panel, rail.

## Shapes

Data is square; containers are softly rounded. Die cells are square rectangles with no radius, tiled in data units so they stay square at any size. Legend swatches are 11px with a 1px corner. Chips round at 4px. Panels, widgets and the readout table round at 6px with overflow clipped. The wafer outline is a thin graphite ring at half opacity with a V notch at the bottom.

## Components

### Buttons
- **Shape:** 6px corners from the theme base radius, hairline widget border.
- **Primary use:** Previous / Next queue stepping and the CSV download, graphite text on panel.
- **Hover / Focus:** Streamlit defaults driven by the graphite primary colour.

### Chips
- **Style:** 0.72rem, 650, 0.04em tracking, 2px 8px, 4px corners, set beside a readout value.
- **State:** warn (amber tint, deep amber ink) and ok (green tint, deep green ink). Status words only.

### Cards / Containers
- **Readout table:** silicon panel, 1px hairline border, 6px corners, rows divided by hairlines, label column 36% wide in muted label type, value cells padded 10px 16px 12px with a value line and a muted 0.82rem note line.

### Inputs / Fields
- **Style:** Streamlit widgets on the panel tone, hairline border always shown, 6px corners; on the rail they sit on the raised rail tone with pale green accent.

### Navigation
- **Sidebar rail:** graphite background, light rail ink, rail rules between groups, pale green slider and toggle accent. Tabs below the map carry live counts.

### Wafer Bin Map (signature)
One square cell per unit on a serpentine probe path in time order, inside a notched ring. Bins: caught failure (red fill), missed failure (red outline), false alarm or queued (graphite fill), not queued (pale green fill). The selected die gets a graphite 2.2px halo square. Moving the budget redraws which cells are queued; clicking a cell selects it for the contribution detail. Below it, a legend of swatches with mono counts and a muted note stating that the map is a batch map, not a spatial one.

### Charts
Altair "wafer" theme: transparent background, Public Sans, muted 11–12px axis text, chart-grid lines, hairline domain and ticks, titles 13px/600 graphite left-anchored, legends at the bottom. Limits and budget markers are graphite dashed rules.

## Do's and Don'ts

### Do:
- **Do** read any batch-level result on the wafer bin map before any table.
- **Do** reserve red, amber and pale green for bin state, and pair each state with a shape, dash or label.
- **Do** set measurements, unit IDs and sensor names in JetBrains Mono at 0.86em.
- **Do** use the readout table (label, tabular value, muted note) for headline numbers, with the random baseline next to every catch count.
- **Do** separate surfaces with tone and 1px hairline (`hairline`) borders.

### Don't:
- **Don't** add a KPI-tile grid; the readout table replaces it.
- **Don't** add drop shadows or elevation; the system is flat.
- **Don't** round die cells; data cells stay square.
- **Don't** introduce a chromatic accent that carries no bin meaning.
