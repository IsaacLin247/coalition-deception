---
title: "Coalition Deception and Defensive Adaptation in a Social-Deduction Benchmark"
author:
  - Isaac Lin
  - Xi Chen
keywords: [multi-agent reinforcement learning, learning dynamics, strategic communication, robust aggregation, partial observability]
date: ""
abstract: |
  How well do defenses against coordinated testimony withstand attackers trained against them?
  We investigate a social-deduction benchmark with structured claims, exact truth labels, and
  two impostors sharing a reinforcement-learning policy. A corrected 240-job study finds that
  learning increases innocent ejections and aligned voting even as directly false claims
  decrease. Defensive adaptation depends on the outcome: with five crew and two impostors,
  multi-round coalition victory falls from 0.967 to 0.360 after defender training and returns
  to 0.851 after coalition retraining, while innocent ejections remain frequent. Ten-generation
  crossplay shows renewed gains against the latest opponent, without establishing an
  asymptotic cycle. A hypothesis-based defender incurs substantial truthful-testimony error
  and is vulnerable to targeted training, which increases false ejection from 0.126 to 0.422.
  Ballot diagnostics identify an abstention-related failure route under fixed recorded votes.
  Mechanism experiments show that alibi errors can increase without coalition-weight
  amplification; a separate Gaussian surrogate characterizes when such amplification is
  possible under explicit assumptions. Dependence penalties have mixed effects on victory
  and voting harm. These findings support evaluating defenses across adaptive opponents,
  distinct outcomes, and directly measured mechanisms.
---

# Introduction

In social deduction, players combine private observations and public testimony to decide
whom to eject. Agreement can reflect independent evidence or coordination within a hidden
coalition. A defense that handles a scripted alibi may therefore perform differently against
an adversary trained to exploit its decisions.

We study this problem in an Among Us-inspired benchmark with two policy-sharing impostors,
structured claims, and explicit records of evidence, testimony, and votes. The benchmark
supports both scripted attacks and learned responses. Our question is how defensive
performance changes with the attacker, the training sequence, and the definition of harm.

Strategic communication connects to cheap-talk and signaling games [@crawford1982;
@lewis1969], trainable sender–receiver systems [@kharitonov2019; @condorelli2024], and
information aggregation with multiple senders [@chen2025; @arieli2023]. Robust aggregation,
Byzantine-resilient reinforcement learning, and peer prediction offer related defenses with
explicit assumptions [@arieli2023; @figura2021; @witkowski2012]; geometric-median mechanisms
also face strategyproofness limits [@elmhamdi2023]. Social-deduction agents and language-model
studies examine deception and hidden roles [@kopparapu2022; @sarkar2025]. Adversarial
communication, credit assignment, cooperation under uncertain incentives, and secret
collusion provide adjacent settings [@blumenkamp2021; @li2021; @orzan2024; @motwani2024].

Adaptive evaluation is already an established requirement in adversarial robustness
[@tramer2020], as is learning approximate responses to policy populations [@lanctot2017].
Our contribution is a comparison within a structured social-deduction benchmark: exact
claim labels separate explicitly false testimony from harmful coordination, crossplay
separates opponent-specific gains from transfer, and complete game records distinguish
coalition victory from innocent ejection. Mechanism studies examine credibility weights,
score construction, and ballots directly. A Gaussian report model supplies a separate,
analytically tractable account of weight amplification.

The results resist a single ranking of defenses. The hypothesis rule improves against some
scripts but makes many errors under supported testimony and targeted attacks. Learned
defenders can suppress coalition victory while retaining frequent innocent ejections.
Repeated response training produces renewed gains against recent opponents, while
dependence penalties can improve one outcome and worsen another. The analysis therefore
keeps these outcomes and mechanisms distinct.

# Methods

## Benchmark and information

The benchmark contains two hidden impostors and 3, 5, or 7 crew members, denoted
3+2, 5+2, and 7+2. Roles are randomized each episode. A meeting consists of
structured claims, responses, and a plurality vote with abstention; tied votes
produce no ejection, and ejection does not reveal roles. The incident time is
public. The two impostors share a policy, reward, and a feature identifying their
partner. Unless ablated, spatial policies also exchange private four-symbol
messages.

The meeting-only configuration (F1–F2) generates private evidence through six
random-walk steps on the Hub–Electrical, Hub–Navigation, and Hub–Medbay star map, with stay probability 0.35 and one
exogenously placed incident. The spatial configuration uses the same rooms plus
an Electrical–Navigation edge, eight movement steps, and two tasks per agent.
An impostor can create an incident only with one crew target and no other living
crew in the room; its partner may be present. Spatial experiments use either one
meeting or up to eight rounds. These configurations differ in evidence generation,
map, and communication, so their comparison does not isolate attack adaptation.

Players see their own evidence and the public transcript. Scripted policies use
their permitted within-round positive and negative sightings; learned spatial
actors receive the corresponding time-indexed private-history block, without
unseen or future positions. Scripted movement uses only available observations.
Role-blind incident masks and legal no-op attempts prevent action availability
from revealing the partner in hidden-identity controls. Claims receive exact
true, unsupported, false, or contradictory labels from recorded states and
speaker evidence. No-information is a separate action, not a truth label.

## Testimony and defenses

Scripted crew choose informative supported claims and use the same suspicion
rule for responses and votes. F2 compares four coalition scripts: supported
testimony; a lone creator lying about its location; a creator's false self-alibi
supported by its partner; and joint scene-placement claims framing an innocent.
The latter two coordinate responses and votes; without a channel, their alibi
rooms need not agree. Supported testimony still
allows selective disclosure and coalition-protective voting. These scripts differ
in several behaviors, rather than manipulating agreement alone.

For voter $v$ and candidate $j$, ordinary rules combine private evidence and
speaker-specific testimony:

$$
S_v(j)=\mathrm{own}_v(j)+\operatorname{AGG}_s[\mathrm{signal}_v(s,j)].
$$

Scene-placement claims and accusations increase suspicion; witness alibis reduce
it. Personally detected lies penalize their speaker within that speaker's row.
The rules use an unweighted sum (called mean), a rescaled median or trimmed mean,
or credibility-weighted sums. Credibility is −3 for a caught lie, otherwise −2
for self-contradiction, otherwise the count of compatible informative speakers.
Softmax temperatures are 1.5 (soft) and 0.15 (sharp).

Dependence-aware credibility multiplies soft weights by $e^{-4\mathrm{ex}_s}$,
where $\mathrm{ex}_s$ measures agreement with a speaker's closest partner beyond
its average agreement. It pools claim compatibility, alibis, supportive responses,
and aligned non-skip votes over three meetings. The ballot variant discounts votes
and remains active with learned crew.

The hypothesis rule enumerates two-member coalitions, excluding publicly known
crew, and averages constraint feasibility over honesty gates and auxiliary
creator/time assignments. Each coalition speaker's informative claims are accepted
with probability one half or ignored. Infeasible hypotheses receive zero weight;
positive scores are normalized. Voting requires membership score above one half,
with lexicographic ties resolved by creator score, private evidence, then agent
index; all-zero views abstain when legal. The solver checks emitted constraints,
with a necessary-constraint relaxation for some negative transition claims.
These are heuristic feasibility scores, not calibrated generative posteriors.
Appendix B lists suspicion coefficients; public configurations specify the
complete actions and policies.

## Training and experimental design

Masked, non-recurrent actor–critic policies use PPO [@schulman2017; @huang2022]
with two 128-unit hidden layers, learning rate $3\times10^{-4}$, discount 1,
GAE parameter 0.95, clip 0.2, four phase-stratified minibatches, and up to eight
epochs. Approximate KL above 0.075 stops an update; the value coefficient and
gradient-norm limit are 0.5, and the entropy coefficient declines from 0.02 to 0.005. Standard stages
train for 400 updates. Each update collects complete batches of 64
episodes in F1 or 32 in spatial experiments. Unfinished trajectories are never
finalized with terminal targets. Evaluation uses a copied policy and independent
action randomness.

Rewards are terminal and do not directly reward lying or coordination. F1 gives
the coalition +1 for an innocent ejection after a real incident and −1 otherwise.
F3's zero-sum balanced-ejection objective gives +1 for innocent ejection, 0 for
no ejection after an incident, and −1 for coalition ejection or no incident.
Single-meeting multigeneration loops retain that coalition objective but train
crew with meeting-ejection rewards after a real incident: +1 for coalition ejection,
−1 for innocent ejection, and 0 for no ejection or no incident. Reward controls compare
these constructions. In multiround survival experiments, the coalition wins when
living impostors are at least as numerous as living crew after a meeting vote;
crew win by eliminating the coalition or reaching the round cap without that
condition. A separate control substitutes the
incident-gated innocent-ejection count for survival.

Response training proceeds $C_0\rightarrow D_1\rightarrow C_1$: coalition against
scripted soft-credibility crew, crew against frozen $C_0$, then coalition against
frozen $D_1$. Every stage starts fresh actor and critic weights. Multigeneration
loops continue these approximate responses with matched seed offsets and budgets.
They do not track a continuously updated policy. Crossplay tests frozen policies;
policy distances use fixed probe observations and include initialization effects.

The corrected study comprises 240 jobs and 1,255 learner stages:

| Experiment | Independent units and evaluation budget |
|:--|:--|
| F1 | Ten training seeds per population; 400 holdout episodes at genuinely untrained and final checkpoints. |
| F2 | Ten evaluation replicates; 1,000 episodes per rule/condition/population. |
| F3–F4 | Ten seeds per population; 1,000 final episodes for each of four pairings. |
| Multigeneration | Ten seeds through ten generations for 5+2 in both regimes and 3+2 multiround; five seeds through three generations for 7+2 multiround. Each stage uses 1,000 episodes; full crossplay uses 500 per cell. |
| Defended loop | Ten seeds and four generations. |
| Adaptive defenses | Ten seeds at 5+2; five at 3+2 for dependence defense and at 7+2 for hypothesis defense. |
| Static spatial studies and controls | Five seeds; information, defender-budget, and reward controls use matched reference seeds. |

Table 1. Study cohorts and evaluation budgets. Episodes and generations are repeated observations within a seed or evaluation replicate; complete job definitions and comparison expressions are versioned with the data.

Controls remove partner identity, the private channel, or both; increase defender
training to 800 or 1,600 updates where specified; and replace single-meeting or
multiround rewards. Complete designs, cohorts, and signed comparisons are versioned
with the data.

## Outcomes, inclusion, and inference

For learned experiments, the independent unit is a training seed; for scripted
experiments, it is an independently seeded evaluation batch. Episodes and
generations are repeated observations. Every planned job passed checks of source,
configuration, checkpoints, and raw records. No seed was removed or stopped for
its outcome. Historical experiments, incomplete runs, and engineering smoke tests
are excluded. Several implementation components changed together; historical
differences cannot identify an isolated repair's effect.

Single-meeting outcomes include false ejection, creator ejection, no ejection,
and incident probability. Creator ejection and survival require an incident and
are not unconditional complements. Multiround outcomes distinguish coalition
victory, any innocent ejection, and mean innocent-ejection count. Whole-game records
include every completed meeting, including incident-free meetings; terminal-meeting
and pooled per-meeting rates remain separate. Dependence records retain the
pre-ejection electorate and count historical meetings once. Conditional metrics
are averaged over seeds with nonzero denominators, retaining contributing counts;
undefined values are not replaced by zero.

Final evaluation starts at seed 1987654321 plus run seed, separately from
monitoring at 987654321 plus run seed, and does not advance training randomness.
Predetermined episode indices are completed without selecting faster games.
F2 pairs replicate identities, while its rule/condition-specific streams do not
hold evidence or complete transcripts fixed.

For 303 specified contrasts in 15 families, matched seed outcomes are subtracted
before calculating mean effects, sample SDs, and pointwise 95% percentile intervals
from 20,000 resamples of complete seed differences. Two-sided exact sign-flip
tests preserve magnitudes and require null sign symmetry/exchangeability;
pairing alone does not establish that assumption. Holm correction applies within
each declared family. All contrasts, including unfavorable and nonsignificant
results, are released. This post-audit specification was not prospective
preregistration of the original research.

The minimum raw two-sided p-value is 0.001953125 with ten seeds and 0.0625 with
five. Thus five-seed tests cannot reject at 0.05, nor can a ten-seed family with
at least 26 tests pass Holm's first threshold. Families were retained despite
this limit. With five seeds, only 126 distinct bootstrap seed-count multisets
are possible; their intervals are descriptive. Intervals are not simultaneous, and
nonrejection does not establish equivalence. Unqualified numerical comparisons
are descriptive means and sample SDs.

## Mechanism diagnostics and response gaps

The descriptive mechanism study evaluates 540,000 episodes: ten replicates
per population, four scripts, and nine rule/variant cells, with 500 episodes each.
Cells within a condition share initial evidence; responses and votes can change.
The descriptive ballot study evaluates retained hypothesis-attack checkpoints
in 135 cells of 500 episodes, using ten seeds at 5+2 and five at 7+2. Removing
ballots holds other recorded votes fixed and does not estimate behavioral adaptation.

For crossplay matrix $M$, the exploratory finite-pool gap
$G_k=\max_iM_{ik}-\min_jM_{kj}$ includes later-trained policies. Simultaneous
Hoeffding bands account for cell uncertainty and selection of extrema, assuming
independent episode draws within each cell. This retrospective gap is neither
true exploitability nor a convergence test.

# Results

## Learning increases harmful coordination

In the meeting-only game (F1), coalition training increases false ejection from 0.045 to
0.271 at 3+2, from 0.073 to 0.2125 at 5+2, and from 0.091 to 0.194 at 7+2. All three
final-minus-untrained contrasts have Holm-adjusted $p=0.0117$ in the six-test F1 family.
Aligned non-skip voting and creator survival also increase, while directly false claims
decrease at every population (Figure 1). At 5+2, aligned voting rises from 0.108 to 0.316,
and the false-claim fraction falls from 0.423 to 0.349. Thus greater voting harm need not
require more explicitly false statements. These joint changes do not isolate communication
as their causal driver.

![](figures/final/f1_learning_and_holdout.pdf){width=100%}

Figure 1. Coalition learning against scripted soft-credibility crew in the meeting-only game. Curves use monitored evaluations; untrained and final points use independent holdout episodes. Variability is across ten seeds, not independent episode replicates.

## Scripted performance and adaptive robustness differ

The seven-rule scripted comparison (F2) exposes an accuracy tradeoff. At 5+2, hypothesis
scoring reduces false ejection relative to the mean under alibi (0.1469 versus 0.3273) and
framing (0.1542 versus 0.4106), but increases it under the lone liar (0.1907 versus 0.1311).
Its truthful-testimony false-ejection rate is 0.3700, whereas mean and soft credibility have
zero observed errors (Figure 2). The corresponding hypothesis errors at 3+2 and 7+2 are
0.2447 and 0.3926. Here *truthful* describes supported claim contents; information selection
and coalition voting remain adversarial.

![](figures/final/f2_tenseed_crew5_rules.pdf){width=100%}

Figure 2. False-ejection rates for seven defenses against four scripted coalitions at 5+2, with ten replicates and 1,000 episodes per cell. Creator-ejection outcomes are available in the released data. These descriptive comparisons retain the full F2 multiplicity families.

Median and trimmed rules can discard useful sparse testimony, so scalar-contamination
robustness does not ensure accurate votes. Hypothesis feasibility is also insufficient:
a bounded replay of ten truthful-testimony errors found that the true coalition retained
positive weight and its timelines satisfied the emitted constraints. Some errors arose
from exact score ties resolved by agent index; another involved unanimous honest abstention.
This replay identifies possible failure routes, not their prevalence. The 63- and 36-test
F2 families cannot reject at 0.05 with the retained ten-seed exact-test resolution.

Targeted training against the hypothesis rule (F8) exposes a further vulnerability. At 5+2,
its false-ejection rate increases from 0.1255 against a soft-trained coalition to 0.4217
against a hypothesis-trained coalition: paired effect 0.2962, pointwise 95% CI
[0.2521, 0.3418], Holm-adjusted $p=0.0273$. The hypothesis-trained/hypothesis-tested pairing
exceeds soft-trained/soft-tested false ejection by 0.2275 ($p=0.0273$). Yet the hypothesis-trained attack obtains only 0.0962 against
mean and 0.0976 against soft credibility (Figure 3). The attack specializes to its training
defense. At 7+2 the hypothesis rate reaches 0.6036, a descriptive result based on five seeds.

![](figures/v2/hypothesis_crossplay.pdf){width=100%}

Figure 3. False-ejection rates in adaptive hypothesis crossplay at 5+2. Rows identify the coalition's training defense and columns its evaluation defense; entries are mean ± sample SD across ten seeds with 1,000 final episodes per cell. The hypothesis-trained coalition transfers poorly to the other two rules.

An independent 67,500-episode ballot study reproduces the elevated hypothesis-attack error:
0.4240 at 5+2 and 0.6048 at 7+2. Honest voters abstain at rates 0.6608 and 0.8988. Removing
coalition ballots prevents 97.26% and 100% of observed false ejections; coalition ballots
alone suffice in 97.75% and 100%. Crew-only false ejection is 0.0142 and zero, respectively.
Table 2 compares attack types at 5+2. These are fixed-ballot counterfactuals: they support an
abstention-related route to failure, but do not predict how agents would vote under a changed
rule or establish abstention as the only cause.

| Attack | FE | Crew skip | Necessary | Sufficient | Crew-only FE |
| :-- | --: | --: | --: | --: | --: |
| mean | 0.075 | 0.196 | 0.651 | 0.603 | 0.032 |
| soft | 0.117 | 0.271 | 0.684 | 0.662 | 0.040 |
| hypothesis | 0.424 | 0.661 | 0.973 | 0.978 | 0.014 |

Table 2. Independent ballot diagnostics against the hypothesis defender at 5+2. Entries average within-seed rates over ten seeds. Necessary and sufficient condition on observed false ejection and hold recorded votes fixed; crew-only FE removes coalition ballots. Full precision, SDs, and denominator counts are released with the data.

## Response training changes victory and voting harm differently

In spatial single-meeting crossplay (F3), the learned defender produces more false ejections
against the fixed initial coalition. At 5+2, rates are 0.1940 for $C_0$–$D_0$, 0.4031 for
$C_0$–$D_1$, and 0.5385 for $C_1$–$D_1$; the adapted coalition obtains only 0.0759 against
$D_0$. The specified defender-effect contrast, $C_0$–$D_0$ minus $C_0$–$D_1$, is negative
at all three populations (adjusted $p=0.0176$). Coalition adaptation against $D_1$ is not
uniformly beneficial: its false-ejection effect is −0.0482 at 3+2, +0.1354 at 5+2, and
+0.1124 at 7+2; the negative effect does not reject.

Defender training also changes incident opportunities. At 5+2, $C_0$ creates incidents in
0.963 of games against $D_0$ but 0.254 against $D_1$. Thus the comparison combines movement,
incident creation, testimony, and voting. Some 3+2 seeds have no incidents against $D_1$;
their conditional rates remain undefined while unconditional summaries retain all ten seeds.

In eight-round play (F4), defender training substantially reduces coalition victory, but
victory and voting harm separate (Figure 4). At 5+2, victory changes from 0.9670
($C_0$–$D_0$) to 0.3603 ($C_0$–$D_1$), then to 0.8514 ($C_1$–$D_1$). The last adaptation
raises victory by 0.4911 while changing the probability of any innocent ejection only from
0.8392 to 0.8467 and reducing mean innocent-ejection count by 0.4249 per game. At 7+2,
victory rises by 0.5104 while mean count falls by 1.0066. Differently terminated or shorter
games can therefore produce more coalition wins with fewer total innocent ejections.

![](figures/final/f4_tenseed_crew5_crossplay.pdf){width=100%}

Figure 4. Eight-round crossplay at 5+2 across ten seeds. Coalition victory, any innocent ejection, mean innocent-ejection count, and terminal outcomes are distinct. Whole-game harm includes incident-free meetings. The 27-contrast F4 family has no Holm rejection at 0.05.

Transfer again depends on population: $C_1$–$D_0$ wins 0.2182 at 5+2 and 0.0630 at 7+2,
but 0.9882 at 3+2. All F4 effects are descriptive under the retained family: its minimum
attainable adjusted p-value is 0.0527. Large point estimates and pointwise intervals do not
override that familywise limitation.

## Repeated responses produce finite, opponent-specific gains

Ten-generation crossplay (F5) shows renewed coalition gains against the latest defender.
In the separate 5+2 multi-round loop, coalition victory is 0.9658 for $C_0$–$D_0$, 0.3524
for $C_0$–$D_1$, and 0.8562 for $C_1$–$D_1$. At generation ten it is 0.3068 for
$C_9$–$D_{10}$ and 0.7968 for $C_{10}$–$D_{10}$, while $C_{10}$–$D_0$ achieves only 0.1804
(Figure 5). These values come from the multi-generation evaluation, independently of F4.

![](figures/v2/finite_crossplay.pdf){width=100%}

Figure 5. Coalition-victory rates in full ten-generation 5+2 crossplay, averaged across ten seeds with 500 episodes per cell. Each diagonal entry minus the entry immediately above measures coalition adaptation against a fixed defender. Poor transfer to earlier defenders distinguishes these gains from a uniformly stronger coalition.

Matrix gain averages $M_{g,g}-M_{g-1,g}$ over response generations: the outcome is
false ejection for single-meeting loops and coalition victory for multi-round loops.
Mean gains are 0.300 in 5+2 single-meeting play, 0.501 in 5+2 multi-round play,
and 0.576 in 3+2 multi-round play. Their tests pass Holm correction in the 25-test dynamics
family ($p=0.0488$). The shorter 7+2 loop has a descriptive gain of 0.418 but only five seeds.
No specified late-minus-early or parity test rejects after correction. Policy distances and
finite-pool response gaps are retained in the released analysis; neither they nor the
observed alternation establish convergence, nonconvergence, optimal responses, or an
attracting cycle. Each stage starts from fresh weights and faces a different frozen opponent.

## Dependence penalties have conditional benefits

Dependence-aware defenses (F7) do not show a general adaptive-attack gain. At 5+2, training
against the testimony-discount rule changes its false-ejection rate from 0.1819 to 0.1696:
effect −0.0123, CI [−0.0516, 0.0374], adjusted $p=1$. Against the combined rule and ballot
discount, retraining changes 0.2044 to 0.1896: effect −0.0148, CI [−0.0513, 0.0215],
adjusted $p=1$. At 3+2, combined-defense error instead rises descriptively from 0.4132 to
0.8722; five seeds preclude a raw 0.05 rejection. None of these comparisons establishes
immunity. The static spatial rule sweep likewise shows attack-dependent rankings and is
distinct from F2's exogenous-evidence comparison.

In the four-generation 5+2 loop, dependence-aware defense increases any innocent ejection
by 0.042425 after coalition stages (CI [0.034775, 0.049400]) and by 0.104375 after defender
stages ([0.073274, 0.135025]); both adjusted p-values are 0.0117. After coalition stages it
reduces coalition victory by 0.028775 ([−0.045625, −0.012050], $p=0.0410$). Table 3 retains
all six comparisons, including the nonsignificant defender-stage victory result. Smaller
adaptation gains therefore coexist with greater voting harm.

| Outcome | Summary | Reference | Defended | Difference | Holm p |
| :-- | --: | --: | --: | --: | --: |
| Any FE | C stages | 0.847 ± 0.009 | 0.890 ± 0.007 | +0.042 | 0.0117 |
| Any FE | D stages | 0.658 ± 0.051 | 0.763 ± 0.047 | +0.104 | 0.0117 |
| Any FE | Matrix gain | 0.191 ± 0.051 | 0.129 ± 0.046 | -0.062 | 0.0234 |
| Coalition win | C stages | 0.817 ± 0.025 | 0.789 ± 0.026 | -0.029 | 0.0410 |
| Coalition win | D stages | 0.309 ± 0.065 | 0.348 ± 0.054 | +0.040 | 0.1387 |
| Coalition win | Matrix gain | 0.509 ± 0.086 | 0.445 ± 0.070 | -0.064 | 0.0469 |

Table 3. Dependence-aware versus reference loops over the same four generations and ten seeds. Values are means ± sample SD; differences are defended minus reference, and stage summaries exclude $C_0$. Holm correction retains the complete six-comparison family.

## Information, budget, and reward controls

The five-seed controls qualify the interpretation. Removing both the private channel and
partner feature gives initial-coalition single-meeting false ejection 0.2518, versus 0.1854
by default. Shared parameters, rewards, and possible role inference remain, so this does
not isolate every coordination mechanism or establish that explicit communication is needed.

On matched seeds, extending defender training from 400 to 800 or 1,600 updates gives
$C_1$–$D_1$ single-meeting false ejection 0.5252, 0.5672, and 0.5160. Alternative rewards
also change descriptive adaptation gains: in the four-generation single-meeting loop,
matrix gains are 0.287 by default, 0.164 with balanced rewards for both sides, and 0.220
with meeting rewards for both. In the two-generation multi-round control, victory gains
are 0.440 under survival and 0.333 under framing count. None of the nine outcome-ablation,
eight budget, or eighteen reward comparisons rejects after Holm correction. Five-seed
resolution and finite budgets prevent conclusions of equivalence or optimal training.

# Mechanisms: score construction and correlated reports

The 540,000-episode mechanism study separates coalition-weight amplification from other
ways credibility weighting can change votes. Under alibi attacks, soft credibility's
mean coalition weight is 0.2630 at 3+2, 0.1425 at 5+2, and 0.0976 at 7+2, below the uniform
shares 0.5000, 0.3333, and 0.2500. These shares count actual reporters after the incident
victim is removed. Across all 240,000 retained honest-voter records, none exceeds its
uniform share at tolerance $10^{-12}$. This result is specific to these scripted records.

Nevertheless, soft credibility increases alibi false ejection relative to the mean by
0.0112, 0.0680, and 0.1586 across the three populations. The score combines two relevant
terms: witness alibis reduce suspicion, while a caught liar's self-incriminating penalty
sits inside that speaker's downweighted row. Removing alibi credit, including the positive
was-with subject credit governed by the same coefficient, reduces alibi errors for every
tested rule and size. Moving the catch penalty into private evidence reduces
soft alibi false ejection by 0.0242, 0.0912, and 0.1750. These interventions match initial
evidence but allow responses and votes to change.

The benefits are conditional. Removing alibi credit increases soft framer error by 0.0876
at 5+2 and 0.1332 at 7+2; moving the catch penalty increases soft lone-liar error by 0.1540
at 3+2. Lower false ejection can also mean more abstention: under 3+2 alibi attacks, every
tested variant has zero creator ejection and coalition-favorable rate one, meaning no
coalition member is ejected, including meetings with no ejection. Under 5+2 lone
liars, soft catch-outside reduces false ejection from 0.1598 to 0.1316 but increases
coalition-favorable outcomes from 0.2000 to 0.2416. Neither excessive total coalition
weight nor a universally beneficial score repair explains these observations.

The Gaussian model (F6) isolates weight amplification under different assumptions. There
are $n$ honest reports with independent variance-$\sigma^2$ Gaussian noise, and $m<n$
coalition reports with bias $b$, marginal variance $\sigma_c^2$, and equicorrelation $\rho$.
Reports within distance $h$ corroborate one another; weights are proportional to
$e^{\gamma c_i}$, where $c_i$ counts corroboration. Replacing counts by their expectations
defines the surrogate coalition weight

$$
W_C=\frac{f}{f+(1-f)e^{-\gamma\Delta c}},\qquad
f=\frac{m}{n+m},\quad \Delta c=\bar c_C-\bar c_H.
$$

For $\gamma>0$, amplification $W_C>f$ occurs exactly when $\Delta c>0$. The median instead
has a uniform error bound in coalition bias when $m<n$. A long-window dependence penalty
can make surrogate coalition weight non-increasing with correlation on a specified branch;
Appendix A gives the assumptions, threshold, and proof. The amplification and dependence
claims concern the surrogate. For example, with $n=5$, $m=2$, equal unit variances, $b=20$, $h=0.43$,
$\rho=\gamma=1$, it predicts $W_C=0.295>2/7$, while 100,000 Monte Carlo trials give mean
realized weight 0.277, below the uniform share.

The numerical report and sensor examples remain illustrative. For the report model at
$n=5,m=2,h=0.4\sigma,b=4\sigma,\sigma_c=\sigma,\gamma=1$, credibility MSE in units of $\sigma^2$ changes from
0.87 with independent coalition noise to 1.84 with perfectly correlated noise. The
dependence-aware rule gives 0.31 in the correlated case but 1.02 in the independent case.
In the twelve-sensor analogue with three coordinated compromised sensors, credibility
RMSE is $0.99\sigma$, versus $0.83\sigma$ for the mean, $0.58\sigma$ for the median, and
$0.56\sigma$ for the dependence variant across twenty runs. These parameter-specific
benefits do not establish a general repair. The model omits the game's alibi credit and
catch penalties, and its reporter-minority condition excludes the 2+2 electorate remaining
after an incident in the initial 3+2 game.

# Discussion and limitations

Three distinctions organize the findings. First, performance against a script does not
determine robustness to a trained opponent. Second, coalition victory and innocent
ejection can move differently. Third, credibility-induced errors need not arise through
excessive total coalition weight. Adaptive crossplay and direct score and ballot records
are needed to distinguish these mechanisms.

The scope is limited by the benchmark and learning procedure. Small maps, structured
claims, and two policy-sharing impostors do not establish behavior in natural-language
deliberation or independently incentivized coalitions. Non-recurrent actors retain limited
cross-round memory. Fresh initialization, finite PPO budgets, and a restricted policy pool
preclude claims about optimal responses or asymptotic dynamics. The hypothesis rule's
deterministic ties, constraint relaxations, and uncalibrated scores also limit what its
vulnerability establishes about stronger inference-based defenses.

Statistical resolution is a material limitation, not evidence that small studies establish
equivalence. The five-seed and large-family restrictions described in Methods apply even
to large point estimates. The mechanism interventions preserve initial evidence rather
than full transcripts; ballot removal preserves votes rather than behavior. These limits
prevent unique causal explanations or universal defense rankings. Several implementation
repairs changed together, so differences from historical runs cannot identify one repair's
effect. The released protocols and raw records make the reported comparisons auditable.

# Conclusion

Coalition learning increases harmful voting without increasing directly false claims.
Hypothesis scoring trades some scripted-attack benefits for substantial baseline error and
targeted vulnerability. Repeated training restores gains against recent opponents within
the observed horizon, while defenses can reduce coalition victory and still increase
innocent ejections. Evaluating adaptive opponents, distinct harms, and measured mechanisms
provides a more precise account of defensive performance than a single win rate or
credibility score.

# Acknowledgements {-}

OpenAI Codex, using GPT-6, assisted with implementation review, correction and testing of
research code, numerical reanalysis, and manuscript editing. Numerical measurements were
produced by the executable experiments and analysis scripts described in this paper.

# Data availability {-}

Code, complete seed-level summaries, all planned comparisons, and diagnostic tables are
available in [Coalition Deception](https://github.com/IsaacLin247/coalition-deception).
The [corrected-study-2026-09-11 release](https://github.com/IsaacLin247/coalition-deception/releases/tag/corrected-study-2026-09-11)
contains the frozen source, checkpoints, raw episode and meeting records, environment
manifests, and checksums. Appendix B describes reproduction. The complete numerical grids
and extended figures remain available there, including comparisons summarized in this
article.

# References {-}

::: {#refs}
:::

# Appendix A. Proposition: corroboration capture under correlated testimony {-}

**Setting.** Let $1\le m<n$, $N=n+m$, $f=m/N$, $\sigma,\sigma_c,h>0$, $\gamma,\lambda\ge0$, and $\rho\in[0,1]$. Honest reports are $x_i=\theta+\varepsilon_i$ with independent $\varepsilon_i\sim\mathcal N(0,\sigma^2)$. Coalition reports are $y_j=\theta+b+\eta_j$, with $\eta\sim\mathcal N(0,\sigma_c^2[(1-\rho)I+\rho\mathbf1\mathbf1^\top])$, independent of the honest noise. Define $r=(x,y)$, $c_i=\#\{k\ne i:|r_i-r_k|\le h\}$, and the realized estimator $\hat\theta_{\rm cred}=\sum_i w_i r_i$, $w_i\propto e^{\gamma c_i}$. Dependence-aware weights additionally multiply by $e^{-\lambda\mathrm{ex}_i}$, where $\mathrm{ex}_i=\max_{k\ne i}D_{ik}-(N-1)^{-1}\sum_{k\ne i}D_{ik}$ and $D_{ik}$ is the agreement frequency across $T$ independent repetitions with fixed model parameters.

**Exact agreement probabilities.** With $s^2=\sigma^2+\sigma_c^2$,
$$
p_{HH}=\operatorname{erf}(h/(2\sigma)),\qquad
p_{CC}(\rho)=\operatorname{erf}\!\left(\frac{h}{2\sigma_c\sqrt{1-\rho}}\right),\quad p_{CC}(1)=1,
$$
$$
p_{HC}=\Phi((h-b)/s)-\Phi((-h-b)/s).
$$
Hence $\bar c_H=(n-1)p_{HH}+mp_{HC}$, $\bar c_C=(m-1)p_{CC}+np_{HC}$ and $\Delta c=\bar c_C-\bar c_H$ is non-decreasing in $\rho$.

**Deterministic mean-field surrogate.** Replace each random corroboration count by its group expectation. The resulting coalition weight and estimator are
$$
W_C=\frac{f}{f+(1-f)e^{-\gamma\Delta c}},\qquad
\tilde\theta=(1-W_C)\bar x+W_C\bar y.
$$
This substitution defines a surrogate. In general $W_C\ne\mathbb E[\sum_{j\in C}w_j]$, and it does not give an exact boundary for the realized estimator. Its exact bias and mean-squared error are
$$
\operatorname{bias}(\tilde\theta)=W_Cb,\qquad
\operatorname{MSE}(\tilde\theta)=W_C^2b^2+(1-W_C)^2\frac{\sigma^2}{n}
+W_C^2\sigma_c^2\left(\rho+\frac{1-\rho}{m}\right).
$$
The approximation $\operatorname{MSE}(\tilde\theta)\approx W_C^2b^2$ requires the displayed variance terms to be negligible relative to $W_C^2b^2$.

**Proposition (surrogate amplification and robustness).**

1. *Weight amplification.* For $\gamma>0$, $W_C>f$ if and only if $\Delta c>0$, equivalently
$$
(m-1)p_{CC}>(n-1)p_{HH}-(n-m)p_{HC}.
$$
At $\gamma=0$, $W_C=f$. When $\Delta c>0$, $W_C$ increases with $\gamma$ and is non-decreasing in $\rho$; $W_C>1/2$ precisely when $\gamma\Delta c>\log((1-f)/f)$. Weight amplification increases the surrogate's squared bias for $b\ne0$; a comparison of total MSE must also include the variance terms above. At $\rho=1$ and negligible cross-group agreement, the amplification condition reduces to $(m-1)>(n-1)p_{HH}$. Negligible cross-group agreement requires separation relative to the noise scale, for example $(|b|-h)/s\gg1$.

2. *Median robustness and the bias reference.* Because $m<n$, the pooled median $M$ lies between the smallest and largest honest reports for every realization, irrespective of the coalition reports. Therefore
$$
\operatorname{MSE}(M)\le\sigma^2\mathbb E\!\left[\max_{1\le i\le n}|Z_i|^2\right]<\infty,
\qquad Z_i\stackrel{\rm iid}{\sim}\mathcal N(0,1),
$$
uniformly in $b$ and $\rho$. For odd $N$, writing $k=(N+1)/2$ and $Z_{(j)}$ for honest order statistics gives the sharper pathwise bound $x_{(k-m)}\le M\le x_{(k)}$, hence $|\operatorname{bias}(M)|\le\sigma\mathbb E[Z_{(k)}]$. As $n,m$ grow with $m/N\to f<1/2$, the latter upper bound tends to $B_\infty=\sigma\Phi^{-1}(1/[2(1-f)])$. Thus $W_C|b|=B_\infty$ defines an asymptotic bias-comparison reference, not a finite-sample MSE break-even boundary. For $0<W^*=B_\infty/|b|<1$, $W_C|b|>B_\infty$ is equivalent to
$$
\gamma\Delta c>\log\!\frac{(1-f)W^*}{f(1-W^*)}.
$$
When $\Delta c>0$ and the right-hand side is positive, division by $\Delta c$ gives the positive reference gain $\gamma^*$. Finite-sample error comparisons use the actual risks or the finite-sample bound above. At $f=1/2$, the median loses its uniform bound against arbitrarily displaced coalition reports; this does not imply that every estimator fails on every distribution.

3. *Dependence penalty.* In the long-window limit, for $m\ge2$,
$$
\overline{\mathrm{ex}}_C=\max(p_{CC},p_{HC})-\frac{(m-1)p_{CC}+np_{HC}}{N-1},
$$
$$
\overline{\mathrm{ex}}_H=\max(p_{HH},p_{HC})-\frac{(n-1)p_{HH}+mp_{HC}}{N-1}.
$$
For $m=1$, $\overline{\mathrm{ex}}_C=0$ and coalition correlation has no effect. The defended surrogate replaces $\gamma\Delta c$ by $\gamma\Delta c-\lambda(\overline{\mathrm{ex}}_C-\overline{\mathrm{ex}}_H)$. On any correlation interval where $m\ge2$ and $p_{CC}(\rho)\ge p_{HC}$, its log-weight advantage depends on $\rho$ through
$$
\left[\gamma(m-1)-\lambda\frac{n}{N-1}\right]p_{CC}(\rho).
$$
Consequently $\lambda\ge\gamma(m-1)(N-1)/n$ makes the defended surrogate weight non-increasing with correlation on that interval; a strict inequality makes it decrease where $p_{CC}$ increases. For $5+2$, the threshold is $1.2\gamma$. If $p_{CC}<p_{HC}$, the coefficient instead is $(m-1)[\gamma+\lambda/(N-1)]$, so the stated repair condition does not apply there. A comparison between $\rho=0$ and $\rho=1$ requires the first branch to hold throughout that interval.

*Proof sketch.* Groupwise softmax normalization gives $W_C$, and independence of the two groups gives the exact surrogate variance. The amplification and majority statements follow by rearranging the logistic expression. Fewer than half the reports are adversarial, so neither central rank can escape the honest range; for odd $N$ at most $m$ insertions can shift the median's honest rank, giving the stated order-statistic bounds. Gaussian symmetry gives the bias bound, and convergence of central order statistics gives its fixed-fraction limit. Finally, the strong law applied to each pair's agreement indicators gives the long-window matrix entries. Taking each row's actual maximum and collecting coefficients in $p_{CC}$ yields the two dependence branches. $\square$

# Appendix B. Implementation and reproduction {-}

The original 82-file executable/configuration snapshot is identified by SHA-256
691f7687435424185064b24a9fee1832c6e7b32b9661efe20a10e35ca0c17d35.
The portable package records its separate fingerprint while preserving the scientific
engine and 240 job designs. Original outputs retain their original protocol identity.
Complete configurations, action definitions, scripts, and analysis commands accompany
the public code and release; the standard PPO and evaluation settings are given in Methods.

For scripted suspicion, private evidence scores scene sightings +3.0, sightings elsewhere
−3.5, unplaced agents +1.0, and non-informative speakers +0.4. Testimony scores a caught
lie +4.0 on its speaker, self-contradiction +3.0, scene placement +1.5, witness alibi −2.0,
self-alibi −0.5, positive was-with claims −1.5 on the subject and −0.5 on the speaker,
accusation +0.8, and contradiction +0.4. Voting requires suspicion above 0.5. Hypothesis
votes use the separate feasibility and tie-breaking procedure in Methods.

The archive retains 11,885 checkpoints, including untrained, intermediate, and final
weights. Checks cover source and configuration identities, finite architecture-compatible
tensors, completed update budgets, matching final checkpoints, raw episode/meeting outcomes,
and exact seed cohorts. All 303 contrasts in 15 families remain available, including those
summarized here. Operating systems and PyTorch builds need not yield bit-identical training.

The mechanism records include 540,000 episodes and 240,000 honest-voter views. The ballot
records include 135 cells, 67,500 episodes, and 317,670 honest score views. Verification
reconstructs the full pre-ejection electorate, abstention, and ballot necessity/sufficiency.
Two terminal-coordination cells in the 7+2 loop are undefined because a coalition voter was
already lost in every game; the primary outcomes remain complete, with no seed removed or
undefined rate replaced by zero. Mean catch-outside differs from its algebraically equal
baseline in two 3+2 alibi episodes because a $2.22\times10^{-16}$ score difference resolves
an argmax tie; this does not support a substantive intervention effect for the mean.

This single-document edition is built from `paper_v2.md` by `build_v2.sh`, which generates
standalone TeX and PDF with Pandoc and XeLaTeX. The accompanying source manifest and verifier
check the retained numerical tables, figure identities, citations, and original-manuscript
preservation. Full numerical grids and extended diagnostics remain in the public analysis.
