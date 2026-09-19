# FML written exam drafting

Migrated from the vault's FML Exam Draft procedure on 2026-09-19. These slides
establish covered material; private exam drafts belong in the separate registered
`fml-exams` repository, never this course-material repository.

Confirm the exam session and locate the private exam repo through the vault's
Repository Registry. If it is unavailable, report that gap; creating a new remote
requires the owner's explicit decision. Read its local instructions when available.

Extract section, subsection and frame titles from this year's `slides/*.tex` and
confirm coverage as of the exam date. Do not assume syllabus or prior-year coverage;
ask about a genuinely uncertain taught boundary.

Read two or three recent private `main_YYYY_MM.tex` exams to match the preamble,
exam-class options, session header and within-year Roman session numbering. Preserve
the established mix (normally 8–10 questions, 2–5 points each, about 28–31 points):
theory, a NumPy/Python coding question and multiple-choice questions with exactly
three choices asking for the correct or incorrect statement. These are examples;
the actual recent exams determine the format and total.

Use confirmed topics and vary the angle from recent sessions. Use `\mathbf{}` for
vectors/matrices rather than the slides' local `\vc{}` macro. Compile the new
`main_YYYY_MM.tex` twice with pdflatex under the private repo's requirements
(including shell-escape only where its trusted source requires it); verify point
totals, errors/overfull boxes and rendered page layout. Prefer layout adjustments
before cutting approved questions. Keep .tex and .pdf; leave temporary compiler
artifacts out of version control.

Show the rendered questions/layout for the owner's explicit review before pushing
the exam. This content-review requirement remains even with standing private-repo
publication approval. Do not publish an exam or its solutions to course slides or
a public site. Completion means confirmed coverage, correct totals, verified PDF
and owner-reviewed exam content in its intended private repository.
