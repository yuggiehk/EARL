# EARL ICML 2026 Project Page

This is a static GitHub Pages project page for:

**EARL: Towards a Unified Analysis-Guided Reinforcement Learning Framework for Egocentric Interaction Reasoning and Pixel Grounding**

The layout is designed to be close to the DINO-R1 / Academic Project Page / Nerfies-style paper website.

## Important

The real paper figures were **not included** in this generated ZIP because only a screenshot of your folder was available in the chat. The HTML is already written to reference your existing figure filenames directly.

Put these files in the same folder as `index.html`:

```text
fig_1.png          # performance radar chart
fig_2.png          # overall EARL architecture
fig_3.png          # feature fusion / AFS figure
fig_4.png          # inference case comparison visualization
supp_fig_1.png     # model inference case
supp_fig_2.png     # model inference case
example_paper.pdf  # local paper PDF, optional because arXiv links are also included
```

## Recommended repository layout

If you deploy from the official code repository `yuggiehk/EARL`, use a `docs/` directory:

```text
EARL/
├── docs/
│   ├── index.html
│   ├── .nojekyll
│   ├── static/
│   ├── fig_1.png
│   ├── fig_2.png
│   ├── fig_3.png
│   ├── fig_4.png
│   ├── supp_fig_1.png
│   ├── supp_fig_2.png
│   └── example_paper.pdf
```

Then enable GitHub Pages:

```text
Settings -> Pages -> Deploy from a branch -> main -> /docs
```

Expected URL:

```text
https://yuggiehk.github.io/EARL/
```

## Quick commands

From your local repo root:

```bash
mkdir -p docs
cp -r /path/to/generated_page/* docs/
cp icml2026/fig_1.png docs/
cp icml2026/fig_2.png docs/
cp icml2026/fig_3.png docs/
cp icml2026/fig_4.png docs/
cp icml2026/supp_fig_1.png docs/
cp icml2026/supp_fig_2.png docs/
cp icml2026/example_paper.pdf docs/

git add docs
git commit -m "Add ICML 2026 project page"
git push
```
