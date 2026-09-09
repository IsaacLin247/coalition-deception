"""Scripted movement is invariant to occupancy outside the agent's observations."""
import numpy as np
import pytest
from social_collusion.config import load_env_config
from social_collusion.env import compute_masks, reset
from social_collusion.env.enums import Phase
from social_collusion.env.observations import last_seen
from social_collusion.policies.registry import make_coalition


def _state():
    cfg = load_env_config('full_short_game').with_(n_agents=5, n_coalition=2, max_rounds=1)
    st = reset(cfg, np.random.default_rng(7))
    st.phase, st.turn, st.horizon = int(Phase.FREE_PLAY), 2, 2
    st.roles[:] = [1,1,0,0,0]
    st.positions[:] = [0,2,1,2,2]
    st.alive[:] = True
    st.incident_time = st.incident_room = st.incident_creator = -1
    st.report_turn = -1
    st.marker_seen[:] = False
    return st


def _action(state, condition, seed=71):
    masks = [head[0] for head in compute_masks(state)]
    out = make_coalition(condition).act_single(state,0,masks,np.random.default_rng(seed))
    assert all(mask[value] for mask,value in zip(masks,out))
    return out


@pytest.mark.parametrize('condition', ['truthful','lone_liar','alibi','framer'])
@pytest.mark.parametrize('after_incident', [False,True])
def test_unseen_adjacent_occupancy_does_not_change_action(condition, after_incident):
    a = _state()
    b = a.copy()
    b.positions[:,2] = 3  # Isolated crew moves between unobserved adjacent rooms.
    if after_incident:
        for st in [a,b]:
            st.incident_time,st.incident_room,st.incident_creator = 1,0,0
            st.marker_seen[0] = True
            st.alive[4] = False
            st.positions[1,4] = 0
            st.positions[2:,4] = -1
        a.positions[:,1:4] = 2
        b.positions[:,1:4] = 3
    assert all(np.array_equal(x,y) for x,y in zip(last_seen(a,0),last_seen(b,0)))
    assert all(np.array_equal(x[0],y[0]) for x,y in zip(compute_masks(a),compute_masks(b)))
    assert np.array_equal(_action(a,condition),_action(b,condition))


def test_partner_unobserved_incident_location_does_not_change_movement():
    a = _state()
    b = a.copy()
    for st,room in [(a,1),(b,3)]:
        st.incident_creator,st.incident_time,st.incident_room = 1,1,room
    assert np.array_equal(_action(a,'alibi'),_action(b,'alibi'))


def test_unknown_partner_roles_do_not_change_sighting_based_movement():
    a = _state()
    a.config = a.config.with_(partner_known=False)
    a.positions[1,0] = 1
    a.positions[1,1] = 1
    b = a.copy()
    b.roles[1],b.roles[2] = b.roles[2],b.roles[1]
    assert all(np.array_equal(x,y) for x,y in zip(last_seen(a,0),last_seen(b,0)))
    assert np.array_equal(_action(a,'alibi'),_action(b,'alibi'))
