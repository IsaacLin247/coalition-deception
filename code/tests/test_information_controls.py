"""Information controls must change retained evidence without revealing hidden state."""

import numpy as np
import pytest

from social_collusion.config import EnvConfig, load_env_config
from social_collusion.env.action_masks import (
    compute_masks,
    incident_attempt_targets,
    valid_incident_targets,
)
from social_collusion.env.enums import Head, Phase, Role, RoomAction
from social_collusion.env.knowledge import UNKNOWN, ObserverKnowledge
from social_collusion.env.observations import build_actor_obs, get_spec, obs_dim
from social_collusion.env.state import DEAD
from social_collusion.env.transition import reset, transition


def spatial_state(**overrides):
    cfg = load_env_config("full_short_game").with_(
        **(dict(n_agents=5, enable_symbol_channel=False, symbol_mode="none") | overrides)
    )
    state = reset(cfg, np.random.default_rng(812), seed=812)
    state.roles[:] = [Role.COALITION, Role.COALITION, Role.CREW, Role.CREW, Role.CREW]
    return state


def decode_private_history(state, observer):
    cfg = state.config
    block = get_spec(cfg).block(build_actor_obs(state, observer), "private_history")
    block = block.reshape(cfg.n_times, 1 + cfg.n_rooms + cfg.n_agents)
    known = np.full((cfg.n_times, cfg.n_agents), UNKNOWN)
    present = np.full(cfg.n_times, DEAD)
    for t, row in enumerate(block):
        if row[0]:
            room = int(np.argmax(row[1:1 + cfg.n_rooms]))
            present[t] = room
            known[t, row[1 + cfg.n_rooms:].astype(bool)] = room
    return known, present


def test_full_history_reconstructs_all_scripted_positional_knowledge():
    state = spatial_state()
    rng = np.random.default_rng(921)
    state.positions[:] = rng.integers(0, state.config.n_rooms, size=state.positions.shape)
    state.turn = state.horizon = 5
    state.phase = int(Phase.CLAIM_ROUND)
    # A death removes later positional knowledge, not earlier sightings.
    state.positions[3:, 4] = DEAD
    for observer in range(state.config.n_agents):
        expected = ObserverKnowledge.build(state, observer)
        known, present = decode_private_history(state, observer)
        np.testing.assert_array_equal(known, expected.known)
        np.testing.assert_array_equal(present, expected.present)
        for t in range(state.config.n_times):
            for agent in range(state.config.n_agents):
                for room in range(state.config.n_rooms):
                    absent = ((known[t, agent] != UNKNOWN and known[t, agent] != room)
                              or (known[t, agent] == UNKNOWN and present[t] == room))
                    assert absent == expected.knows_absent(agent, t, room)


def test_history_preserves_an_earlier_sighting_erased_by_last_seen():
    state = spatial_state(actor_private_history=False)
    state.positions[:] = 3
    state.positions[:, 0] = 0
    state.positions[:, 1] = 0
    state.positions[1, [0, 1]] = 1
    state.turn = state.horizon = 3
    state.phase = int(Phase.CLAIM_ROUND)
    alternate = state.copy()
    alternate.positions[1, 1] = 2  # not seen at t=1; seen again at t=2 and t=3
    np.testing.assert_array_equal(build_actor_obs(state, 0), build_actor_obs(alternate, 0))
    state.config = state.config.with_(actor_private_history=True)
    alternate.config = state.config
    assert not np.array_equal(build_actor_obs(state, 0), build_actor_obs(alternate, 0))
    assert decode_private_history(state, 0)[0][1, 1] == 1
    assert decode_private_history(alternate, 0)[0][1, 1] == UNKNOWN


@pytest.mark.parametrize("reported", [False, True])
def test_full_history_ignores_unseen_rooms_and_future_positions(reported):
    state = spatial_state(partner_known=False)
    state.positions[:] = 2
    state.positions[:, 0] = 0
    state.turn = 3
    state.horizon = 3 if reported else 0
    state.phase = int(Phase.CLAIM_ROUND if reported else Phase.FREE_PLAY)
    alternate = state.copy()
    alternate.positions[:4, 1:] = 3  # both are outside the observer's room
    alternate.positions[4:] = 1  # even the observer's unelapsed locations are hidden
    alternate.roles[[1, 2]] = alternate.roles[[2, 1]]
    np.testing.assert_array_equal(build_actor_obs(state, 0), build_actor_obs(alternate, 0))
    known, present = decode_private_history(state, 0)
    assert (known[4:] == UNKNOWN).all()
    assert (present[4:] == DEAD).all()
    assert (known[:4, 0] == 0).all()


def test_legacy_layout_is_unchanged_and_new_spatial_configs_enable_history():
    legacy = load_env_config("full_short_game").with_(actor_private_history=False)
    legacy_dict = legacy.to_dict()
    del legacy_dict["actor_private_history"]
    assert not EnvConfig(**legacy_dict).actor_private_history
    full = legacy.with_(actor_private_history=True)
    assert load_env_config("full_short_game").actor_private_history
    assert not load_env_config("meeting_only").actor_private_history
    for name, block in get_spec(legacy).slices.items():
        assert get_spec(full).slices[name] == block
    assert obs_dim(full) - obs_dim(legacy) == full.n_times * (1 + full.n_rooms + full.n_agents)


def test_hidden_partner_masks_are_invariant_to_other_agents_roles():
    state = spatial_state(partner_known=False)
    state.turn = 1
    state.positions[:] = [0, 0, 0, 2, 3]
    alternate = state.copy()
    alternate.roles[[1, 3]] = alternate.roles[[3, 1]]
    assert valid_incident_targets(state, 0, 1).any()
    assert not valid_incident_targets(alternate, 0, 1).any()  # two crew would witness
    for original, changed in zip(compute_masks(state), compute_masks(alternate)):
        np.testing.assert_array_equal(original[0], changed[0])
    np.testing.assert_array_equal(build_actor_obs(state, 0), build_actor_obs(alternate, 0))
    assert set(np.flatnonzero(incident_attempt_targets(state, 0, 1))) == {1, 2}


def test_hidden_partner_mask_does_not_reveal_an_incident_elsewhere():
    state = spatial_state(partner_known=False)
    state.turn = 2
    state.positions[:] = [0, 0, 1, 2, 3]
    alternate = state.copy()
    alternate.incident_creator = 1
    alternate.incident_time = 1
    alternate.incident_room = 3
    for original, changed in zip(compute_masks(state), compute_masks(alternate)):
        np.testing.assert_array_equal(original[0], changed[0])
    assert compute_masks(state)[Head.ROOM_ACTION][0, RoomAction.CREATE_INCIDENT]


def test_known_partner_incident_masks_retain_exact_feasibility():
    state = spatial_state(partner_known=True)
    state.turn = 1
    state.positions[:] = [0, 0, 0, 2, 3]
    assert set(np.flatnonzero(incident_attempt_targets(state, 0, 1))) == {2}
    np.testing.assert_array_equal(incident_attempt_targets(state, 0, 1), valid_incident_targets(state, 0, 1))
    state.positions[:, 3] = 0
    assert not incident_attempt_targets(state, 0, 1).any()


@pytest.mark.parametrize("positions,target,success", [
    ([0, 0, 0, 2, 3], 1, False),  # partner target is a legal, unsuccessful attempt
    ([0, 2, 0, 0, 3], 2, False),  # a second crew witness prevents the incident
    ([0, 0, 0, 2, 3], 2, True),   # one crew and the partner can share the room
])
def test_hidden_partner_attempts_resolve_without_invalid_action_feedback(positions, target, success):
    state = spatial_state(partner_known=False)
    state.turn = 1
    state.positions[:] = positions
    masks = compute_masks(state)
    action = np.stack([m.argmax(axis=1) for m in masks], axis=1)
    action[:, Head.MOVE] = positions
    action[:, Head.ROOM_ACTION] = RoomAction.WAIT
    action[0, Head.ROOM_ACTION] = RoomAction.CREATE_INCIDENT
    action[0, Head.INCIDENT_TARGET] = target
    result = transition(state, action, np.random.default_rng(300), validate=True).state
    assert result.invalid_action_attempts == 0
    assert (result.incident_time >= 0) == success
    if success:
        assert result.incident_victim == target
        assert not result.alive[target]
    else:
        assert result.alive.all()
        assert not any("invalid" in event.type.lower() or "failed" in str(event.payload).lower()
                       for event in result.events)


def symbol_step(state, symbol=2, seed=792):
    state.phase = int(Phase.SYMBOL)
    masks = compute_masks(state)
    action = np.stack([m.argmax(axis=1) for m in masks], axis=1)
    action[state.alive & (state.roles == Role.COALITION), Head.SYMBOL] = symbol
    rng = np.random.default_rng(seed)
    result = transition(state, action, rng, validate=True).state
    return result, rng


def test_hidden_partner_inbox_contains_payload_without_sender_identity():
    state = spatial_state(partner_known=False, enable_symbol_channel=True, symbol_mode="learned")
    alternate = state.copy()
    alternate.roles[[1, 3]] = alternate.roles[[3, 1]]
    result, rng = symbol_step(state)
    changed, changed_rng = symbol_step(alternate)
    assert result.symbols_received[0] == changed.symbols_received[0] == 2
    np.testing.assert_array_equal(build_actor_obs(result, 0), build_actor_obs(changed, 0))
    np.testing.assert_array_equal(rng.integers(0, 1000, 8), changed_rng.integers(0, 1000, 8))


@pytest.mark.parametrize("mode", ["learned", "random"])
def test_hidden_partner_death_never_removes_inbox_payload(mode):
    state = spatial_state(partner_known=False, enable_symbol_channel=True, symbol_mode=mode)
    state.alive[1] = False
    state.positions[:, 1] = DEAD
    for seed in range(12):
        result, _ = symbol_step(state, seed=seed)
        assert 0 <= result.symbols_received[0] < state.config.k_symbols
        symbol_obs = get_spec(state.config).block(build_actor_obs(result, 0), "symbol")
        assert symbol_obs[-1] == 0  # missing-inbox category is never delivered


def test_hidden_channel_cover_randomness_does_not_depend_on_partner_survival():
    state = spatial_state(partner_known=False, enable_symbol_channel=True, symbol_mode="learned")
    state.alive[1] = False
    state.positions[:, 1] = DEAD
    alternate = state.copy()
    alternate.roles[[1, 2]] = alternate.roles[[2, 1]]
    result, rng = symbol_step(state)
    changed, changed_rng = symbol_step(alternate)
    # Both worlds have exactly the same visible deaths; only the hidden partner
    # role differs. Constant cover draws also keep subsequent environment RNG
    # unchanged in the learned-symbol condition.
    assert result.symbols_received[0] >= 0
    assert changed.symbols_received[0] == 2
    np.testing.assert_array_equal(rng.integers(0, 1000, 8), changed_rng.integers(0, 1000, 8))


def test_known_partner_channel_preserves_missing_inbox_after_partner_death():
    state = spatial_state(partner_known=True, enable_symbol_channel=True, symbol_mode="learned")
    state.alive[1] = False
    state.positions[:, 1] = DEAD
    result, _ = symbol_step(state)
    assert result.symbols_received[0] == -1
