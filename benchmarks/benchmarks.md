read uploads: 0.000 seconds
07_Addenda_and_Tender_Query_Responses.pdf: 47 statements
02_Particular_Conditions_of_Contract.pdf: 101 statements
01_Instructions_to_Tenderers.pdf: 150 statements
03_Employers_Requirements.pdf: 235 statements
06_Drawing_Notes_and_Tender_Programme.pdf: 165 statements
05_MEP_Specification.pdf: 172 statements
04_Architectural_Specification.pdf: 339 statements
ingest: 1200.233 seconds
/home/mc/.pyenv/versions/heyaudio_env/lib/python3.10/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
clustering: 15.862 seconds
Found 41 contradiction(s)
--- Contradiction ---
[31] 07_Addenda_and_Tender_Query_Responses.pdf, page 4 | Revision P02: Drawing A-301 Elevations is reissued at revision P02 to correct the window reference numbers on the east elevation. No change to the extent or performance of the windows.
[539] 06_Drawing_Notes_and_Tender_Programme.pdf, page 2 | Drawing A-301: Drawing A-301: Elevations, Scale 1:200, Rev P01, Date 2026-10-03
Reason: Statement [31] indicates that Drawing A-301 has been reissued at revision P02 to correct window reference numbers, while statement [539] lists Drawing A-301 as Rev P01 dated 2026-10-03. These cannot both be true — the revision level has changed from P01 to P02, indicating an update, which contradicts the static P01 revision in [539].
--- Contradiction ---
[837] 05_MEP_Specification.pdf, page 6 | Pass-through dishwasher DW-01 / connected load: Connected load: 9.5 kW
[838] 05_MEP_Specification.pdf, page 6 | Pass-through dishwasher DW-01 / electrical supply: Electrical supply: 3N 400 V, 16 A
Reason: The connected load of 9.5 kW at 3N 400 V implies a current draw of approximately 13.7 A (using P = √3 × V × I × PF, assuming PF ≈ 0.9), which is less than the 16 A rated supply. However, the specification of 16 A supply does not contradict the 9.5 kW load — it is actually sufficient. Therefore, no actual contradiction exists; the values are compatible. Re-evaluating: no contradiction.
contradiction detection: 183.408 seconds
total: 1399.503 seconds


❯ python llm_speed.py
png: 65.157 seconds, 5 pages, 5299 characters
pdf: 251.642 seconds, 5644 characters


With docling
first: 8.178 seconds
second: 1.353 seconds, 5839 characters

CPU
❯ python docling_server_speed.py
first: 26.558 seconds (server: 23.871)
second: 26.416 seconds (server: 23.047), 5836 characters
~/Development/aecom_challenge on   main
❯ python docling_server_speed.py
first: 24.712 seconds (server: 22.383)
second: 24.602 seconds (server: 22.763), 5836 characters

MANAGED
❯ python watsonx_speed.py 
watsonx: 26.947 seconds, 5269 characters
{'completed_at': '2026-10-26T20:02:21.066Z', 'location': ['speed_test/01_Club_Rules_and_Requirements_watsonx.md'], 'number_pages_processed': 5, 'running_at': '2026-10-26T20:02:09.998Z', 'status': 'completed', 'total_pages': 5}
~/Development/aecom_challenge on   main
❯ python watsonx_speed.py
watsonx: 23.659 seconds, 5269 characters
{'completed_at': '2026-10-26T20:02:59.449Z', 'location': ['speed_test/01_Club_Rules_and_Requirements_watsonx.md'], 'number_pages_processed': 5, 'running_at': '2026-10-26T20:02:50.306Z', 'status': 'completed', 'total_pages': 5}


 python3 liteparse_speed.py
[liteparse] extract: 77.4ms (5 pages)
[liteparse] ocr: 6378.8ms
[liteparse] project: 1.1ms
[liteparse] markdown: 1.1ms
[liteparse] total: 6458.4ms
first: 6.461 seconds
[liteparse] extract: 4.4ms (5 pages)
[liteparse] ocr: 2370.0ms
[liteparse] project: 1.2ms
[liteparse] markdown: 1.1ms
[liteparse] total: 2376.8ms
second: 2.379 seconds, 5 pages, 4901 characters


❯ python3 liteparse_server_speed.py
first: 5.321 seconds
second: 4.328 seconds, 5 pages, 88 characters


❯ python3 llamaparse_speed.py
llamaparse (cost_effective): 9.844 seconds, 5 pages, 6709 characters


❯ python liteparse_server_no_ocr_speed.py
first: 0.911 seconds
second: 0.596 seconds, 5 pages, 4953 characters
