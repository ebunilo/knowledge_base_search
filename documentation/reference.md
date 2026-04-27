# Reference: `vitis_hls_ug_2.pdf` (ingested corpus)

This note describes the **actual** manual at  
`data/sample_docs/public/vitis_hls_ug_2.pdf` so demos and evals can be grounded in the right edition.

## Document identity (from the PDF)

| Field | Value |
|--------|--------|
| **Title** | Vitis High-Level Synthesis User Guide |
| **Document ID** | UG1399 |
| **Version** | **(v2025.2)** |
| **Footer date** | January 22, 2026 |
| **Page count** | 895 |
| **Vendor** | AMD Adaptive Computing (Xilinx legacy flow names still appear in places) |

Earlier informal references in this repo used an older **UG1399 (v2022.2)** outline. **This file is a newer release:** expect differences in **IDE naming** (Vitis **unified** IDE), **licensing**, **supported OS** tables, **“Changed behavior”** migration notes, and **Programmers Guide** structure (e.g. paradigms named *Producer-Consumer*, *Streaming Data*, *Pipelining*; extra material on **dataflow** regions, PIPOs, FIFOs, stream-of-blocks).

## Major sections (abbreviated TOC)

- **Section I — Introduction**  
  Navigating by design process, **supported operating systems**, **Obtaining a Vitis HLS License**, **Changed behavior** (release deltas), benefits of HLS, **Introduction to Vitis HLS components**, **Refactoring C++ source code for HLS**, tutorials and examples.
- **Section II — HLS Programmers Guide**  
  Design principles (three paradigms for **programmable logic**), abstract parallel programming model (control/data-driven tasks, dataflow, PIPOs/FIFOs, stream-of-blocks, **stable arrays**), **Loops** (pipelining, rewind, flush, dependencies), **Arrays**, **Functions**, **Data types** (incl. arbitrary precision), unsupported C/C++ constructs, **Interfaces**, efficient designs, **optimization / troubleshooting** (incl. scheduling, AXI, area).
- **Section III — Using Vitis HLS**  
  Launching the **Vitis unified IDE / flow**, new project, **C simulation**, **C synthesis**, analysis viewers, **pragmas/directives**, **C/RTL co-simulation**, **export RTL**, **command line** / Tcl.
- **Section IV+**  
  Command reference, optimization directives, pragma reference, etc.

(The PDF’s built-in outline has hundreds of sub-entries; retrieval will surface the right subsections by query.)

## How this differs from older UG1399 snapshots

If you previously tested against **v2022.2**-style text:

- Terminology and **IDE** flows target the **unified** Vitis environment and **2025.x** behavior tables.
- **Licensing** and **workspace / metadata** changes are documented explicitly; good for “what changed this release?” questions.
- The **Programmers Guide** includes **deeper dataflow** topics (e.g. WYSIWYG style, nested dataflow pitfalls, memory channels) that older demos might not have highlighted.

## Suggested demo questions (aligned to v2025.2)

Use these for **search** or **Ask** after ingestion:

1. What design processes does this guide tie to (e.g. hardware / IP / platform development), and where do C simulation, C synthesis, and C/RTL co-simulation fit?
2. What are the **supported operating systems** for Vitis HLS on Linux and Windows?
3. How do I **obtain a Vitis HLS license** and what tool features require it (e.g. Code Analyzer, Dataflow viewer)?
4. What **changed behavior** is called out for the **Vitis unified IDE** or workspace **version.ini** when upgrading?
5. What are the **three paradigms for programmable logic** in Chapter 1, and how do **producer–consumer**, **streaming data**, and **pipelining** differ?
6. What is a **dataflow region** coding style, and what are **WYSIWYG** vs **nested dataflow** pitfalls?
7. How can arrays be specified as **PIPOs**, **FIFOs**, or **stream-of-blocks**, and when are **stable arrays** relevant?
8. How does **loop pipelining** work, including **rewind**, **flushing**, and **pipeline types**?
9. What is the flow for **C/RTL co-simulation** in Vitis HLS, and what viewers help debug schedule or dataflow?
10. How do I run **Vitis HLS from the command line**, and which Tcl steps correspond to C simulation, synthesis, and co-simulation?

## Ingestion reminder

Corpus updates are **not** automatic: after adding or replacing the PDF, run (from your deployment environment):

`kb ingest --source src_sample_public --stage index`

(Adjust service/container form as in your `docker compose` setup.)

---

*Generated from the PDF’s metadata, text sample, and outline (`get_toc()`); for sub-page citations always trust the retriever’s cited passage over this summary.*
