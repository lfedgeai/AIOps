# prompt-comparison Specification

## Purpose
TBD - created by archiving change jev-prompt-variants. Update Purpose after archive.
## Requirements
### Requirement: Compare variants side by side
`python -m tier3 compare <dir> [<dir> ...]` and `python -m tier2 compare <dir> [<dir> ...]` SHALL group records by `(engine, prompt name, prompt version)` and print one column per group. Rows SHALL be: detect, RCA, diagnosis, remediation correct, FP runs, FP polls, detection precision, recall and F1, censored runs, invalid replies and engine calls. `python report.py --compare` SHALL label columns `engine / prompt`. `--by-prompt` SHALL group Tier 1 records by prompt instead of by file. Each figure SHALL be computed from the records.

#### Scenario: Three variants
- **WHEN** replay records exist for `baseline`, `strict-incident` and `evidence-options` on the same source records
- **THEN** the report prints three columns, for example a detection-precision row `18/22 (82%) | 18/19 (95%) | 18/24 (75%)`

### Requirement: Counts with denominators
Every cell SHALL show the count, its denominator and a percentage rounded to a whole number, as `k/n (p%)`. A cell with a zero denominator SHALL show `n/a`. F1 SHALL show as `2TP/(2TP+FP+FN)` and a two-place decimal.

#### Scenario: Empty denominator
- **WHEN** a group has no fault runs
- **THEN** its detect cell shows `n/a`

### Requirement: Compare on the same source records
Groups SHALL be compared over the source run IDs they share. If the sets differ, the report SHALL print a warning listing the differing run IDs, and SHALL compute every figure over the common set only. Live and replay records SHALL NOT be put in one group.

#### Scenario: Different live batches, asked for explicitly
- **WHEN** `--unpaired` is given for the live Jev and live Opus Tier 4 batches, which share no observations
- **THEN** each column uses all of its own records, and the report warns that the columns did not see the same observations

#### Scenario: Different live batches without the option
- **WHEN** the same two batches are compared without `--unpaired`
- **THEN** every cell shows `n/a`, and the warning says they share no source records

#### Scenario: One replay stopped early
- **WHEN** the strict-incident replay covers 38 of the 40 records the baseline replay covers
- **THEN** the report warns, names the two missing run IDs, and compares all groups over the 38

### Requirement: Dev and confirm sets
The compare commands SHALL accept `--dev <dir>` and `--confirm <dir>`. They SHALL print the confirm table first, headed "Confirm set (not used to write variants)", then the dev table. A variant whose `dev_records` overlap the confirm set SHALL be flagged in the report.

#### Scenario: Variant written on the confirm set
- **WHEN** a variant's `dev_records` lists `results/tier4/6.2-combined` and that directory is passed as `--confirm`
- **THEN** the report flags that variant's confirm column as "written on this set"

### Requirement: Same-day baseline
When a comparison includes replayed `baseline` records and also the original live records of the same sources, it SHALL show both baseline columns. It SHALL compare variants with the replayed baseline, and say that it does so.

#### Scenario: Jev drift
- **WHEN** replayed `baseline` gives 18/21 detection precision and the original live records gave 18/22
- **THEN** both columns are shown, and the variant deltas are against the replayed 18/21

### Requirement: Mismatched variant hashes
The report SHALL stop with an error when two records share a prompt name and version but have different `sha256` values.

#### Scenario: Hash conflict
- **WHEN** two directories both contain `strict-incident` v1 records with different hashes
- **THEN** the report exits with an error naming the variant, both hashes and one record from each

