# Protocol v4 statistical resolution

This is a post-hoc statistical description of the seven valid Protocol-v4
pairs. It does not alter Protocol v4, reclassify invalid pairs, define a new
benchmark protocol, authorize another run, or authorize a PASS tag. Ratios are
always `Phase3A3 time / Phase3A4 time`.

## Valid paired observations

| Pair | Order | Phase 3A3 ms/op | Phase 3A4 ms/op | Ratio |
|---:|:---:|---:|---:|---:|
| 1 | A3 → A4 | 0.0574737862 | 0.0574756637 | 0.9999673332 |
| 3 | A3 → A4 | 0.0574451871 | 0.0575153437 | 0.9987802119 |
| 4 | A4 → A3 | 0.0575123820 | 0.0574948825 | 1.0003043671 |
| 6 | A4 → A3 | 0.0563706513 | 0.0574595574 | 0.9810491739 |
| 7 | A3 → A4 | 0.0574786253 | 0.0574495587 | 1.0005059495 |
| 8 | A4 → A3 | 0.0576000325 | 0.0576136261 | 0.9997640561 |
| 10 | A4 → A3 | 0.0575753395 | 0.0574995745 | 1.0013176611 |

The arithmetic mean is `0.9973841076`, the median is `0.9999673332`, and the
geometric mean is `0.9973613545`.

## Log-ratio inference

For the seven natural-log ratios:

- mean: `-0.0026421328`
- sample standard deviation: `0.0073123502`
- standard error: `0.0027638086`
- degrees of freedom: 6
- two-sided 95% Student-t interval: `[-0.0094049288, 0.0041206632]`
- back-transformed ratio interval: `[0.9906391592, 1.0041291648]`

The interval's lower bound is only about 0.064 percentage points above the
`0.99` gate. A percentile bootstrap of paired log means, using seed `20260721`
and 100,000 resamples, gives `[0.9917710248, 1.0005262407]`.

## Leave-one-pair-out sensitivity

Each row uses the six remaining pairs and a new two-sided 95% Student-t
interval.

| Removed pair | Geometric mean | Ratio interval |
|---:|---:|:---|
| 1 | 0.9969276855 | [0.9886859061, 1.0052381691] |
| 3 | 0.9971250743 | [0.9888087024, 1.0055113911] |
| 4 | 0.9968716949 | [0.9886590777, 1.0051525329] |
| 6 | 1.0001062997 | [0.9992209497, 1.0009924342] |
| 7 | 0.9968382171 | [0.9886447541, 1.0050995839] |
| 8 | 0.9969614660 | [0.9887038180, 1.0052880818] |
| 10 | 0.9967034915 | [0.9886003208, 1.0048730808] |

The formal seven-pair conclusion is not leave-one-out robust. Removing six of
the seven ordinary near-one ratios moves the lower bound below `0.99`;
removing the anomalous pair 6 instead collapses the interval tightly around
one. Pair 6 therefore dominates both the estimated center and uncertainty.

## Order description

| Order | n | Arithmetic mean | Median | Geometric mean | 95% ratio interval |
|:---|---:|---:|---:|---:|:---|
| A3 → A4 | 3 | 0.9997511649 | 0.9999673332 | 0.9997509049 | [0.9975596224, 1.0019470009] |
| A4 → A3 | 4 | 0.9956088146 | 1.0000342116 | 0.9955729406 | [0.9801398347, 1.0112490534] |

The A4→A3 group contains pair 6 and is consequently far more dispersed. With
only three and four observations this is descriptive evidence, not proof of
an order effect.

## Estimated sample sizes

The estimates iterate the two-sided 95% Student-t critical value using the
observed log-ratio standard deviation. A relative half-width `h` is represented
as `log(1+h)`.

| Desired relative interval half-width | Required valid pairs | Approximate total pairs at observed 7/10 pair validity |
|---:|---:|---:|
| 1.0% | 5 | 8 |
| 0.5% | 11 | 16 |
| 0.25% | 36 | 52 |

If the true center is exactly the observed geometric mean `0.9973613545`, the
estimated count needed for the lower bound of a two-sided 95% interval to
exceed `0.99` is seven valid pairs. At the observed 70% complete-pair validity,
that corresponds to approximately ten total pairs. This formal requirement is
already met, but only narrowly and without leave-one-out robustness.

## Assessment

The 1% gate is statistically decidable with a realistic number of pairs under
the fitted log-ratio model: the formal estimate is seven valid pairs. For a
less brittle decision, at least 11 valid pairs (0.5% half-width), or roughly 16
total pairs at the observed validity rate, is the more defensible minimum.
Achieving a 0.25% half-width would require about 36 valid or 52 total pairs.

More pairs in the same environment would narrow the model-based interval, but
they would not explain the isolated pair-6 shift or the earlier stationarity
failures. Before treating any expanded run as release evidence, a separate
environment qualification without the desktop compositor is recommended.
That qualification should establish stationarity and order symmetry first;
this report deliberately does not specify its protocol.
