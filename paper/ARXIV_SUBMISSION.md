# arXiv submission checklist

## Before you submit

- [ ] Add affiliation and contact email under `\author{...}` in `main.tex` (search for `TODO(author)`).
- [ ] Replace `USERNAME` in the Code Availability URL with the GitHub account that hosts the repo, and push the repo first.
- [ ] Read the whole paper and confirm every claim. The Acknowledgments say an AI assistant helped draft text and code and that the author reviewed it; keep that sentence true.
- [ ] First-time submitters in cs need an endorsement from an existing arXiv author: https://info.arxiv.org/help/endorsement.html
- [ ] Recompile and check: `latexmk -pdf main.tex` (0 warnings at the time of packaging).

## Upload

Upload `openreflect-arxiv-source.tar.gz` (contains `main.tex`, `references.bib`, `main.bbl`). arXiv compiles it with pdfLaTeX.

## Metadata

- **Title:** OpenReflect: A Fully Specified Recipe and Reference Implementation for Training Long-Horizon Self-Improving Agents
- **Authors:** Dulguun Enkhtsatsral
- **Abstract:** paste `arxiv_abstract.txt` (1,746 characters; the limit is 1,920).
- **Primary category:** cs.LG (Machine Learning)
- **Cross-lists:** cs.AI, cs.CL
- **Comments:** 13 pages, 1 figure, 4 tables. Code: https://github.com/USERNAME/OpenReflect
- **License:** CC BY 4.0 (recommended for reuse), or the arXiv non-exclusive license.

## After it is announced

- Put the arXiv ID in the README citation (`note = {arXiv identifier to be added after submission}`).
