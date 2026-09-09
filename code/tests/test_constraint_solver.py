"""Exact search regressions that fail with the old cutoff/arc-only solver."""
from itertools import product
import numpy as np
from social_collusion.config import load_env_config
from social_collusion.env.claims import satisfiable


def test_contradiction_with_more_than_six_agents_is_not_skipped():
    cfg = load_env_config('meeting_only').with_(n_agents=9)
    cons = [('eq_const', j, 1, 0) for j in range(9)]
    assert satisfiable(cons, cfg)
    assert not satisfiable(cons + [('eq_const', 8, 1, 1)], cfg)


def test_joint_correlations_survive_temporal_propagation():
    cfg = load_env_config('meeting_only').with_(rooms=('A', 'B'), edges=())
    # No agent can move. Equality at one time and inequality later are incompatible,
    # although every variable's unary room domain is {A, B} at both times.
    cons = [('eq_pair', 0, 1, 0), ('neq_pair', 0, 1, 2)]
    assert not satisfiable(cons, cfg)
    assert satisfiable(cons[:1] + [('eq_pair', 0, 1, 2)], cfg)


def test_reachability_over_unmentioned_times_is_enforced():
    cfg = load_env_config('meeting_only').with_(rooms=('A','B','C','D'), edges=(('A','B'),('B','C'),('C','D')))
    assert not satisfiable([('eq_const', 0, 0, 0), ('eq_const', 0, 2, 3)], cfg)
    assert satisfiable([('eq_const', 0, 0, 0), ('eq_const', 0, 3, 3)], cfg)


def test_exact_solver_matches_exhaustive_small_worlds():
    cfg = load_env_config('meeting_only').with_(rooms=('A','B','C'), edges=(('A','B'),('B','C')), free_play_turns=2)
    worlds = []
    for values in product(range(3), repeat=6):
        world = np.array(values).reshape(2,3)
        if all(world[a,t+1] == world[a,t] or world[a,t+1] in cfg.neighbors(world[a,t])
               for a in range(2) for t in range(2)):
            worlds.append(world)
    pool = [(kind,a,t,r) for kind in ['eq_const','neq_const'] for a in range(2) for t in range(3) for r in range(3)]
    pool += [(kind,0,1,t) for kind in ['eq_pair','neq_pair'] for t in range(3)]
    def true(world, c):
        if c[0] == 'eq_const': return world[c[1],c[2]] == c[3]
        if c[0] == 'neq_const': return world[c[1],c[2]] != c[3]
        if c[0] == 'eq_pair': return world[c[1],c[3]] == world[c[2],c[3]]
        return world[c[1],c[3]] != world[c[2],c[3]]
    rng = np.random.default_rng(734)
    for _ in range(60):
        cons = [pool[i] for i in rng.choice(len(pool), size=int(rng.integers(1,9)), replace=False)]
        expected = any(all(true(world,c) for c in cons) for world in worlds)
        assert satisfiable(cons,cfg) == expected, cons
