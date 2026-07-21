# Protocol v5 withdrawn before execution

Status: `WITHDRAWN_BEFORE_EXECUTION`

Protocol v5 was predeclared in commit `a4ea3ae` but was never executed. There
are no Protocol-v5 measurements, raw benchmark files, or result files, and no
PASS tag was created.

Before execution, its requirement of 16 valid complete pairs from 20 total
pairs was identified as statistically unsuitable. Protocol v4 observed 17
valid processes among 20 processes, a process-validity estimate of `0.85`.
Under a simplified independence assumption, a complete pair would be valid
with probability `0.85² = 0.7225`; 20 total pairs would therefore yield only
`20 × 0.7225 = 14.45` valid pairs in expectation, below the required 16.

Commit `a4ea3ae` is retained and is not rewritten or deleted. Protocols v1–v4,
production code, and all existing reports remain unchanged. This withdrawal
does not define a replacement protocol and authorizes neither a benchmark run
nor a PASS tag.
