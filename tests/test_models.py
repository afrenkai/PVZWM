import torch

from pvzwm.models.delta import PixelDeltaWorldModel
from pvzwm.models.dynamics import DeterministicDynamics, DynamicsEnsemble
from pvzwm.models.flow import RepresentationFlowWorldModel
from pvzwm.models.jepa import ActionConditionedJEPA, ConvEncoder, VectorEncoder
from pvzwm.models.lewm import ActionConditionedLeWorldModel
from pvzwm.models.regularizers import sigreg_regularizer, vicreg_regularizer
from pvzwm.models.rssm import RecurrentStateSpaceModel
from pvzwm.models.token import CausalTokenWorldModel, PixelFSQTokenizer
from pvzwm.models.visreg import visreg_regularizer


def test_dynamics_shapes_and_loss() -> None:
    model = DeterministicDynamics(20, 7, hidden_dim=32)
    state = torch.randn(8, 20)
    action = torch.randint(0, 7, (8,))
    prediction = model(state, action)
    assert prediction.next_state.shape == state.shape
    torch.testing.assert_close(prediction.next_state, state)
    losses = model.loss(
        state,
        action,
        torch.randn(8),
        torch.randn(8, 20),
        torch.zeros(8),
        state_weight=10.0,
        state_change_weight=5.0,
    )
    assert torch.isfinite(losses["loss"])
    assert 0 <= losses["changed_fraction"] <= 1

    ensemble = DynamicsEnsemble(3, 20, 7, hidden_dim=32)
    ensemble_prediction = ensemble(state, action)
    assert ensemble_prediction.variance is not None
    assert ensemble_prediction.variance.shape == state.shape
    ensemble_losses = ensemble.loss(
        state, action, torch.randn(8), torch.randn(8, 20), torch.zeros(8)
    )
    ensemble_losses["loss"].backward()
    assert torch.isfinite(ensemble_losses["loss"])


def test_vector_dynamics_learns_legality_from_state() -> None:
    model = DeterministicDynamics(
        20, 7, hidden_dim=32, predict_legality=True
    )
    state = torch.randn(8, 20)
    action = torch.randint(0, 7, (8,))
    legal = torch.rand(8, 7) > 0.5
    prediction = model(state, action)
    assert prediction.legal_logits is not None
    assert prediction.legal_logits.shape == legal.shape
    losses = model.loss(
        state,
        action,
        torch.randn(8),
        torch.randn(8, 20),
        torch.zeros(8),
        legal,
    )
    losses["loss"].backward()
    assert losses["legal"] > 0
    assert model.legal_mask(state).shape == legal.shape


def test_vector_dynamics_trains_open_loop_episode_windows() -> None:
    batch, horizon, state_dim, action_dim = 4, 5, 6, 7
    observations = torch.randn(batch, horizon, state_dim)
    next_observations = torch.randn(batch, horizon, state_dim)
    actions = torch.randint(0, action_dim, (batch, horizon))
    rewards = torch.randn(batch, horizon)
    dones = torch.zeros(batch, horizon)
    legal = torch.rand(batch, horizon, action_dim) > 0.5

    model = DeterministicDynamics(
        state_dim, action_dim, hidden_dim=16, predict_legality=True
    )
    losses = model.loss(
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        legal,
    )
    expected_persistence = torch.nn.functional.smooth_l1_loss(
        observations[:, :1].expand_as(next_observations), next_observations
    )
    torch.testing.assert_close(losses["state"], expected_persistence)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])

    ensemble = DynamicsEnsemble(
        3, state_dim, action_dim, hidden_dim=16, predict_legality=True
    )
    ensemble_losses = ensemble.loss(
        observations,
        actions,
        rewards,
        next_observations,
        dones,
        legal,
    )
    ensemble_losses["loss"].backward()
    assert torch.isfinite(ensemble_losses["loss"])


def test_visreg_is_finite_and_differentiable() -> None:
    embeddings = (0.001 * torch.randn(32, 16)).requires_grad_()
    losses = visreg_regularizer(embeddings, slices=8)
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0

    views = (0.01 * torch.randn(3, 12, 16)).requires_grad_()
    view_losses = visreg_regularizer(views, slices=8)
    view_losses.total.backward()
    assert torch.isfinite(view_losses.total)
    assert views.grad is not None
    assert torch.isfinite(views.grad).all()


def test_sigreg_and_vicreg_are_finite_and_differentiable() -> None:
    for regularizer in (
        lambda value: sigreg_regularizer(value, slices=8),
        lambda value: vicreg_regularizer(value).total,
    ):
        embeddings = (0.01 * torch.randn(32, 16)).requires_grad_()
        loss = regularizer(embeddings)
        loss.backward()
        assert torch.isfinite(loss)
        assert embeddings.grad is not None
        assert torch.isfinite(embeddings.grad).all()
        assert embeddings.grad.abs().sum() > 0


def test_jepa_and_rssm_shapes() -> None:
    encoder = VectorEncoder(20, latent_dim=12, hidden_dim=24)
    model = ActionConditionedJEPA(encoder, 7, latent_dim=12, hidden_dim=24)
    state = torch.randn(16, 20)
    action = torch.randint(0, 7, (16,))
    losses = model.loss(
        state, action, torch.randn(16), torch.randn(16, 20), torch.zeros(16), slices=4
    )
    assert torch.isfinite(losses["loss"])
    for regularizer in ("sigreg", "vicreg", "none"):
        alternative = model.loss(
            state,
            action,
            torch.randn(16),
            torch.randn(16, 20),
            torch.zeros(16),
            regularizer=regularizer,
            slices=4,
        )
        assert torch.isfinite(alternative["loss"])

    rssm = RecurrentStateSpaceModel(20, 7, deterministic_dim=16, stochastic_dim=4)
    initial = rssm.initial(3, torch.device("cpu"))
    posterior, prior = rssm.observe(initial, torch.zeros(3, dtype=torch.long), torch.randn(3, 20))
    decoded, reward, done = rssm.decode(posterior)
    assert prior.mean.shape == (3, 4)
    assert decoded.shape == (3, 20)
    assert reward.shape == done.shape == (3,)

    sequence_losses = rssm.loss(
        torch.randn(3, 5, 20),
        torch.randint(0, 7, (3, 5)),
        torch.randn(3, 5),
        torch.randn(3, 5, 20),
        torch.zeros(3, 5),
    )
    sequence_losses["loss"].backward()
    assert torch.isfinite(sequence_losses["loss"])

    pixel_rssm = RecurrentStateSpaceModel(
        (3, 48, 80), 7, deterministic_dim=16, stochastic_dim=4, hidden_dim=24
    )
    pixel_initial = pixel_rssm.initial(2, torch.device("cpu"))
    pixel_posterior, _ = pixel_rssm.observe(
        pixel_initial,
        torch.zeros(2, dtype=torch.long),
        torch.rand(2, 3, 48, 80),
    )
    pixel_decoded, pixel_reward, pixel_done = pixel_rssm.decode(pixel_posterior)
    assert pixel_decoded.shape == (2, 3, 48, 80)
    assert pixel_reward.shape == pixel_done.shape == (2,)
    pixel_losses = pixel_rssm.loss(
        torch.rand(2, 3, 3, 48, 80),
        torch.randint(0, 7, (2, 3)),
        torch.randn(2, 3),
        torch.rand(2, 3, 3, 48, 80),
        torch.zeros(2, 3),
    )
    pixel_losses["loss"].backward()
    assert torch.isfinite(pixel_losses["loss"])


def test_spatial_conv_encoder_retains_configured_grid() -> None:
    encoder = ConvEncoder(3, latent_dim=12, pool_height=3, pool_width=5)
    encoded = encoder(torch.randn(4, 3, 96, 128))
    assert encoded.shape == (4, 12)
    assert encoder.project.in_features == 128 * 3 * 5


def test_lewm_trains_sequences_and_rolls_with_bounded_history() -> None:
    model = ActionConditionedLeWorldModel(
        ConvEncoder(3, latent_dim=16, pool_height=2, pool_width=2),
        action_dim=7,
        latent_dim=16,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        max_history=3,
        projection_head=True,
        prediction_projection_head=True,
        projection_hidden_dim=32,
    )
    observations = torch.rand(4, 3, 3, 32, 48)
    next_observations = torch.rand(4, 3, 3, 32, 48)
    actions = torch.randint(0, 7, (4, 3))
    losses = model.loss(
        observations,
        actions,
        torch.randn(4, 3),
        next_observations,
        torch.zeros(4, 3),
        slices=4,
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert "visreg" in losses
    assert model.projector is not None
    assert model.prediction_projector is not None

    with torch.inference_mode():
        history = model.encode(observations[:, :1])
        step = model.next(history, actions[:, :1])
    assert step.embedding.shape == (4, 16)
    assert step.reward.shape == step.done_logit.shape == (4,)


def test_lewm_frozen_residual_predictor_starts_at_persistence() -> None:
    source_model = ActionConditionedLeWorldModel(
        ConvEncoder(3, latent_dim=8, pool_height=1, pool_width=1),
        action_dim=5,
        latent_dim=8,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        max_history=2,
        projection_head=True,
        projection_hidden_dim=12,
    )
    model = ActionConditionedLeWorldModel(
        ConvEncoder(3, latent_dim=8, pool_height=1, pool_width=1),
        action_dim=5,
        latent_dim=8,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        max_history=2,
        projection_head=True,
        projection_hidden_dim=12,
        residual_prediction=True,
    )
    model.load_representation_state(source_model.state_dict())
    model.freeze_representation()
    model.train()
    assert not model.encoder.training
    assert model.projector is not None and not model.projector.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    with torch.inference_mode():
        embedding = model.encode(torch.rand(3, 2, 3, 24, 32))
        prediction = model.predict(embedding, torch.zeros(3, 2, dtype=torch.long))
    torch.testing.assert_close(prediction.embedding, embedding)


def test_causal_token_world_model_trains_and_generates_pixels() -> None:
    model = CausalTokenWorldModel(
        (3, 32, 48),
        action_dim=7,
        vocab_size=16,
        code_dim=8,
        token_height=2,
        token_width=3,
        tokenizer_hidden_dim=16,
        model_dim=24,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_transitions=4,
    )
    observation = torch.rand(2, 4, 3, 32, 48)
    next_observation = torch.rand(2, 4, 3, 32, 48)
    actions = torch.randint(0, 7, (2, 4))
    losses = model.loss(
        observation,
        actions,
        torch.randn(2, 4),
        next_observation,
        torch.zeros(2, 4),
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert 0 <= losses["codebook_utilization"] <= 1

    model.eval()
    with torch.inference_mode():
        context = model.initial_context(observation[:, 0])
        step = model.generate_next(context, actions[:, 0])
    assert step.tokens.shape == (2, 6)
    assert step.observation.shape == (2, 3, 32, 48)
    assert step.reward.shape == step.done_logit.shape == (2,)


def test_fsq_tokenizer_and_world_model_use_implicit_vocabulary() -> None:
    tokenizer = PixelFSQTokenizer(
        (3, 16, 24), levels=(3, 4), token_height=2, token_width=2, hidden_dim=16
    )
    encoded = tokenizer.encode(torch.rand(2, 3, 16, 24))
    assert tokenizer.vocab_size == 12
    assert encoded.tokens.shape == (2, 4)
    assert int(encoded.tokens.min()) >= 0
    assert int(encoded.tokens.max()) < 12
    assert tokenizer.decode_tokens(encoded.tokens).shape == (2, 3, 16, 24)

    model = CausalTokenWorldModel(
        (3, 16, 24),
        action_dim=5,
        quantizer="fsq",
        fsq_levels=(3, 4),
        token_height=2,
        token_width=2,
        tokenizer_hidden_dim=16,
        model_dim=16,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_transitions=2,
    )
    losses = model.loss(
        torch.rand(2, 2, 3, 16, 24),
        torch.randint(0, 5, (2, 2)),
        torch.randn(2, 2),
        torch.rand(2, 2, 3, 16, 24),
        torch.zeros(2, 2),
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert losses["codebook"] == 0


def test_representation_flow_world_model_trains_and_generates_pixels() -> None:
    model = RepresentationFlowWorldModel(
        (3, 32, 48),
        action_dim=7,
        code_dim=8,
        latent_height=2,
        latent_width=3,
        autoencoder_hidden_dim=16,
        model_dim=24,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    observation = torch.rand(4, 3, 32, 48)
    action = torch.randint(0, 7, (4,))
    losses = model.loss(
        observation,
        action,
        torch.randn(4),
        torch.rand(4, 3, 32, 48),
        torch.zeros(4),
        torch.ones(4, 7, dtype=torch.bool),
        visreg_slices=4,
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert losses["legal"] > 0
    model.eval()
    generated = model.generate_next(observation[:2], action[:2], steps=2)
    assert generated.observation.shape == (2, 3, 32, 48)
    assert generated.reward.shape == generated.done_logit.shape == (2,)
    assert model.legal_mask(generated.latent).shape == (2, 7)

    source = model.encode(observation[:2])
    noisy = torch.zeros_like(source)
    time = torch.full((2,), 0.5)
    dropped = torch.ones(2, dtype=torch.bool)
    velocity_a = model.velocity(
        noisy, source, torch.tensor([0, 1]), time, action_drop_mask=dropped
    )
    velocity_b = model.velocity(
        noisy, source, torch.tensor([5, 6]), time, action_drop_mask=dropped
    )
    torch.testing.assert_close(velocity_a, velocity_b)


def test_pixel_delta_world_model_starts_at_persistence_and_trains() -> None:
    model = PixelDeltaWorldModel(
        (3, 16, 24), action_dim=7, hidden_channels=16, residual_blocks=1
    )
    observation = torch.rand(4, 3, 16, 24)
    action = torch.randint(0, 7, (4,))
    initial = model(observation, action)
    torch.testing.assert_close(initial.observation, observation)

    next_observation = observation.clone()
    next_observation[:, :, 4:8, 6:12] = 1.0 - next_observation[:, :, 4:8, 6:12]
    legal = torch.rand(4, 7) > 0.5
    losses = model.loss(
        observation,
        action,
        torch.randn(4),
        next_observation,
        torch.zeros(4),
        legal,
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert losses["changed_pixel"] > 0
    assert losses["legal"] > 0
