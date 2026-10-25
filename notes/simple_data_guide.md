# Happy Paws practice pack

Ingest **only the three PDFs in `simple_data/`**. This guide is an answer key; do not index it or the generator.

The fictional pack compresses the original seven-document roles into two base documents (rules/requirements and specifications/programme) plus one cumulative addenda document. It retains A4 pages, cover/revision tables, running headers and page numbers, numbered clauses, room and equipment tags, schedules, dated addenda, informational queries, acknowledgement and visit records. It is not a page-for-page rewrite of all seven originals. Each PDF has 5 pages, counting its cover as page 1.

These are invented practice rules, not animal-care guidance. Original `data/` and `case.pdf` are unchanged. The application has not been reconfigured.

## Rebuild

From the project root, run:

```sh
/usr/bin/python3 notes/generate_simple_data.py
```

Requires ReportLab (`reportlab`); the system Python used to generate these files already has it. No application dependencies are changed.

## Expected findings

**9 findings: 4 uncorrected conflicts, 1 omission, 2 corrected conflicts, and 2 ordinary amendments.** The ordinary amendments are the cat bowl count and question deadline; they are not unresolved contradictions. The same nine findings are listed in `expected_conflicts.md`.

Abbreviations: **R** = `01_Club_Rules_and_Requirements.pdf`; **S** = `02_Rooms_and_Equipment_Specification.pdf`; **A** = `03_Addenda_and_Query_Responses.pdf`. Page numbers below are PDF page numbers. Severity is a suggested ranking, not an objective label.

| Topic | Evidence locations | Expected interpretation | Category / suggested severity |
|---|---|---|---|
| Maximum dogs in D-01 | R p3 §4.1 and p5 §8: 6; S p2 §2.1: 8; A p2 Item 1.1: 5; A p3 Item 2.1: 4 | Original numeric conflict, then two changes. Current maximum is **4**; the values 6, 8 and 5 are superseded, not three active rules. | numeric + superseded text / high |
| BW-D bowl capacity | R p4 §5.3 and p5 §8: 2 litres; S p3 §4.1: 2 ml; A p2 Item 1.2: 2 litres | Factor-of-1,000 unit conflict, explicitly corrected. Current capacity **2 litres each**, count **3**. Addendum 2 does not cancel this correction. | unit + superseded text / high |
| Cat bowl count | R p4 §5.3 and p5 §8; S p3 §4.1: 2; A p3 Item 2.2: 3 | Base documents agree but are superseded. Current count **3**; capacity stays **500 ml**. | superseded text / low |
| G-C01 gate height | R p4 §5.1 and p5 §8: 120 cm; S p2 §3.1: 120 cm; S p3 §3.2: 90 cm | Unamended internal text/table conflict and cross-document conflict. **120 cm** wins. G-D01 is consistent at 120 cm. | numeric / high |
| Scratching-post supplier | R p4 §6 and p5 §8: Club Team; S p2 §2.2 and p3 §4.2: Owners; A p4 Q-04: Owners | Unamended scope conflict. **Club Team** wins. Informational Q-04 cannot change responsibility. Count remains 2. | scope / medium |
| Cleaning frequency | R p4 §7: exactly twice, before and after opening hours; S p4 §5.1: exactly once, after closing; A p4 Q-03: once | Unamended conflict. **Twice daily** wins. A later query is not a binding amendment. | numeric / medium |
| Setup duration and deadline | R p3 §3.1: 14 days, 1 May to 15 May; S p4 §6: sequential 5 + 6 + 5 = 16 days, finish 17 May; A p4 Q-05 | Arithmetic in the programme is correct, but the programme misses the deadline by **2 days** and also extends beyond the 16 May opening. Deadline remains **15 May 2027**. Do not invent an addendum extending it. | date/arithmetic / high |
| Missing FA-01 checklist row | R p5 §9 requires a row for every safety item, explicitly including FA-01; S p3 §4.2 supplies FA-01; S p5 §7 claims a complete checklist but has no FA-01 row | Genuine omission from a complete list, not absence from an arbitrary retrieved chunk. The box is supplied; its **checklist row** is missing. R p4 §7 also requires a daily check. | omission / medium |
| Question deadline | R p3 §3: 20 April; A p3 Item 2.3: 25 April | Explicit amendment, not an unresolved conflict. Current deadline **25 April 2027**. | superseded text / low |

Resolved/superseded findings should remain distinguishable from unamended discrepancies, even when precedence tells us which statement governs. Do not describe an unamended discrepancy as a formally corrected document.

## Deliberate non-conflicts

- R p2 §1.2, S p2 §1.2 and A p4 Q-01: Puppy Room = Dog Room = D-01; Kitten Room = Cat Room = C-01. No additional rooms or age restrictions.
- R p4 §5.3, S p3 §4.1 and A p4 Q-02: **500 ml = 0.5 litres**. Cat bowl capacities agree.
- R p4 §5.2 prefers blue dog beds but permits other colours; S p2 §2.1 permits green. Preference versus permission is not a contradiction.
- Six dog beds versus four allowed dogs is not a conflict. A p3 Item 2.1 explicitly leaves bed counts unchanged. Spare beds are allowed.
- A p2 Item 1.3 and p5 Part E list different dates for two separate visits; this is not a date conflict.
- R p3 §4.3 and S p4 §5.2 agree on 17:00 collection and no overnight service.

## Sample questions

1. How many dogs can use the club now? Show the full sequence of changes.
2. Did Addendum 2 undo the dog bowl correction in Addendum 1?
3. Who supplies the cat scratching posts, and does Q-04 override the rules?
4. What is wrong with the cat gate schedule? Where else is its height stated?
5. Are 500 ml and 0.5 litres contradictory?
6. Does the setup programme meet the deadline? Show your arithmetic.
7. Which required safety item is missing from the complete checklist?
8. What changed for cats, and what stayed the same?
9. Can the club use green dog beds?
10. Show the full cleaning clause and all statements that disagree with it.

For a full-pack scan, evaluate quotes, document/page/section citations, entity identification, category, severity, confidence and resolution status separately, as requested in `case.pdf`. Confidence scores should come from your system, not this guide. Test follow-up retrieval of the full clause and repeated mentions as well as single-turn answers.
