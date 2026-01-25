"""
Unit tests for Modular Encoder Architecture

Tests:
1. Encoder creation and forward pass
2. Freeze/unfreeze functionality
3. Add encoder with fusion weight transfer
4. Partial checkpoint loading
5. ONNX export compatibility

Run tests:
    pytest python/tests/test_modular_encoder.py -v
    pytest python/tests/test_modular_encoder.py -v -k "test_add_encoder"
"""

import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.modular_encoder import (
    EncoderModuleConfig,
    ModularEncoderConfig,
    ModularEncoder,
    create_phase_b_config,
    create_lane_encoder_config,
)
from src.models.modular_policy import (
    ModularDrivingPolicy,
    ModularPolicyConfig,
    create_modular_policy_config_phase_b,
)


class TestEncoderModuleConfig:
    """Test EncoderModuleConfig dataclass"""

    def test_basic_config(self):
        config = EncoderModuleConfig(
            name="test",
            input_dim=10,
            hidden_dims=[32, 32],
            output_dim=64,
        )
        assert config.name == "test"
        assert config.input_dim == 10
        assert config.hidden_dims == [32, 32]
        assert config.output_dim == 64
        assert config.frozen == False

    def test_frozen_config(self):
        config = EncoderModuleConfig(
            name="test",
            input_dim=10,
            hidden_dims=[32],
            output_dim=32,
            frozen=True,
        )
        assert config.frozen == True


class TestModularEncoderConfig:
    """Test ModularEncoderConfig dataclass"""

    def test_phase_b_config(self):
        config = create_phase_b_config()

        assert len(config.encoders) == 5
        assert "ego" in config.encoders
        assert "history" in config.encoders
        assert "agents" in config.encoders
        assert "route" in config.encoders
        assert "speed" in config.encoders

        # Check total input dim: 8 + 40 + 160 + 30 + 4 = 242
        assert config.total_input_dim == 242

        # Check total encoder output: 64 + 64 + 128 + 64 + 32 = 352
        assert config.total_encoder_output_dim == 352

    def test_encoder_order(self):
        config = create_phase_b_config()
        assert config.encoder_order == ["ego", "history", "agents", "route", "speed"]


class TestModularEncoder:
    """Test ModularEncoder class"""

    @pytest.fixture
    def encoder(self):
        config = create_phase_b_config()
        return ModularEncoder(config)

    def test_creation(self, encoder):
        assert len(encoder.encoder_modules) == 5
        assert encoder.total_input_dim == 242
        assert encoder.output_dim == 512

    def test_forward_pass(self, encoder):
        batch_size = 4
        obs = torch.randn(batch_size, 242)

        features = encoder(obs)

        assert features.shape == (batch_size, 512)

    def test_slice_indices(self, encoder):
        indices = encoder._slice_indices

        assert indices["ego"] == (0, 8)
        assert indices["history"] == (8, 48)
        assert indices["agents"] == (48, 208)
        assert indices["route"] == (208, 238)
        assert indices["speed"] == (238, 242)

    def test_freeze_encoder(self, encoder):
        # Initially all should be trainable
        assert not encoder.encoder_modules["ego"].is_frozen

        encoder.freeze_encoder("ego")

        assert encoder.encoder_modules["ego"].is_frozen

        # Check gradients are disabled
        for param in encoder.encoder_modules["ego"].parameters():
            assert param.requires_grad == False

    def test_unfreeze_encoder(self, encoder):
        encoder.freeze_encoder("ego")
        encoder.unfreeze_encoder("ego")

        assert not encoder.encoder_modules["ego"].is_frozen

        for param in encoder.encoder_modules["ego"].parameters():
            assert param.requires_grad == True

    def test_freeze_all_encoders(self, encoder):
        encoder.freeze_all_encoders()

        for name, module in encoder.encoder_modules.items():
            assert module.is_frozen

    def test_get_encoder_status(self, encoder):
        status = encoder.get_encoder_status()

        assert len(status) == 5
        assert status["ego"]["input_dim"] == 8
        assert status["ego"]["output_dim"] == 64
        assert status["ego"]["frozen"] == False

    def test_get_trainable_params(self, encoder):
        all_params = list(encoder.parameters())
        trainable_params = encoder.get_trainable_params()

        # All params should be trainable initially
        assert len(trainable_params) == len(all_params)

        # Freeze one encoder
        encoder.freeze_encoder("ego")
        trainable_params = encoder.get_trainable_params()

        # Should have fewer trainable params
        assert len(trainable_params) < len(all_params)


class TestAddEncoder:
    """Test dynamic encoder addition"""

    @pytest.fixture
    def encoder(self):
        config = create_phase_b_config()
        return ModularEncoder(config)

    def test_add_encoder_basic(self, encoder):
        original_num_encoders = encoder.num_encoders
        original_input_dim = encoder.total_input_dim

        lane_config = create_lane_encoder_config()
        encoder.add_encoder(lane_config, freeze_existing=False)

        assert encoder.num_encoders == original_num_encoders + 1
        assert encoder.total_input_dim == original_input_dim + 12
        assert "lane" in encoder.encoder_modules

    def test_add_encoder_freeze_existing(self, encoder):
        lane_config = create_lane_encoder_config()
        encoder.add_encoder(lane_config, freeze_existing=True)

        # Existing encoders should be frozen
        for name in ["ego", "history", "agents", "route", "speed"]:
            assert encoder.encoder_modules[name].is_frozen

        # New encoder should NOT be frozen
        assert not encoder.encoder_modules["lane"].is_frozen

    def test_add_encoder_forward_pass(self, encoder):
        lane_config = create_lane_encoder_config()
        encoder.add_encoder(lane_config, freeze_existing=True)

        # New input dimension: 242 + 12 = 254
        batch_size = 4
        obs = torch.randn(batch_size, 254)

        features = encoder(obs)

        assert features.shape == (batch_size, 512)

    def test_add_encoder_slice_indices(self, encoder):
        lane_config = create_lane_encoder_config()
        encoder.add_encoder(lane_config, freeze_existing=True)

        indices = encoder._slice_indices

        # Lane should be at the end
        assert indices["lane"] == (242, 254)

    def test_add_encoder_fusion_weight_transfer(self, encoder):
        # Get initial fusion weights
        initial_weights = encoder.fusion.mlp[0].weight.clone()

        lane_config = create_lane_encoder_config()
        encoder.add_encoder(lane_config, freeze_existing=True)

        # New fusion should have more input features
        new_weights = encoder.fusion.mlp[0].weight

        # Old dimension weights should be preserved (roughly)
        old_dim = initial_weights.shape[1]
        transferred = new_weights[:, :old_dim]

        # Check weights are similar (small difference due to transfer)
        assert torch.allclose(transferred, initial_weights, atol=1e-5)

    def test_add_encoder_duplicate_name_error(self, encoder):
        with pytest.raises(ValueError, match="already exists"):
            encoder.add_encoder(EncoderModuleConfig(
                name="ego",  # Already exists
                input_dim=10,
                hidden_dims=[32],
                output_dim=32,
            ))


class TestCheckpointLoading:
    """Test partial checkpoint loading"""

    @pytest.fixture
    def saved_encoder(self):
        config = create_phase_b_config()
        encoder = ModularEncoder(config)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save({'model_state_dict': encoder.state_dict()}, f.name)
            return encoder, f.name

    def test_load_encoder_weights(self, saved_encoder):
        original_encoder, checkpoint_path = saved_encoder

        # Create new encoder
        config = create_phase_b_config()
        new_encoder = ModularEncoder(config)

        # Randomize weights
        for param in new_encoder.parameters():
            nn.init.uniform_(param)

        # Load weights
        loaded = new_encoder.load_encoder_weights(
            torch.load(checkpoint_path, weights_only=False)['model_state_dict']
        )

        assert set(loaded) == set(["ego", "history", "agents", "route", "speed"])

        # Verify weights match
        for name in loaded:
            orig_state = original_encoder.encoder_modules[name].state_dict()
            new_state = new_encoder.encoder_modules[name].state_dict()
            for key in orig_state:
                assert torch.allclose(orig_state[key], new_state[key])

        # Cleanup
        Path(checkpoint_path).unlink()

    def test_load_encoder_weights_selective(self, saved_encoder):
        original_encoder, checkpoint_path = saved_encoder

        config = create_phase_b_config()
        new_encoder = ModularEncoder(config)

        # Load only some encoders
        loaded = new_encoder.load_encoder_weights(
            torch.load(checkpoint_path, weights_only=False)['model_state_dict'],
            encoder_names=["ego", "history"]
        )

        assert set(loaded) == set(["ego", "history"])

        Path(checkpoint_path).unlink()

    def test_load_fusion_weights(self, saved_encoder):
        original_encoder, checkpoint_path = saved_encoder

        config = create_phase_b_config()
        new_encoder = ModularEncoder(config)

        success = new_encoder.load_fusion_weights(
            torch.load(checkpoint_path, weights_only=False)['model_state_dict']
        )

        assert success

        # Verify fusion weights match
        orig_state = original_encoder.fusion.state_dict()
        new_state = new_encoder.fusion.state_dict()
        for key in orig_state:
            assert torch.allclose(orig_state[key], new_state[key])

        Path(checkpoint_path).unlink()


class TestModularDrivingPolicy:
    """Test ModularDrivingPolicy class"""

    @pytest.fixture
    def policy(self):
        config = create_modular_policy_config_phase_b()
        return ModularDrivingPolicy(config)

    def test_creation(self, policy):
        assert policy.total_obs_dim == 242
        assert policy.num_parameters > 0

    def test_forward(self, policy):
        batch_size = 4
        obs = torch.randn(batch_size, 242)

        output = policy(obs)

        assert 'action' in output
        assert 'value' in output
        assert output['action'].shape == (batch_size, 2)
        assert output['value'].shape == (batch_size, 1)

    def test_get_action_and_value(self, policy):
        batch_size = 4
        obs = torch.randn(batch_size, 242)

        action, log_prob, value = policy.get_action_and_value(obs)

        assert action.shape == (batch_size, 2)
        assert log_prob.shape == (batch_size, 1)
        assert value.shape == (batch_size, 1)

    def test_evaluate_actions(self, policy):
        batch_size = 4
        obs = torch.randn(batch_size, 242)
        actions = torch.randn(batch_size, 2)

        log_prob, entropy, value = policy.evaluate_actions(obs, actions)

        assert log_prob.shape == (batch_size, 1)
        assert entropy.shape == (batch_size, 1)
        assert value.shape == (batch_size, 1)

    def test_add_encoder(self, policy):
        original_obs_dim = policy.total_obs_dim

        lane_config = create_lane_encoder_config()
        policy.add_encoder(lane_config, freeze_existing=True)

        assert policy.total_obs_dim == original_obs_dim + 12

        # Verify forward pass works
        batch_size = 4
        obs = torch.randn(batch_size, 254)
        output = policy(obs)

        assert output['action'].shape == (batch_size, 2)

    def test_get_trainable_params(self, policy):
        lane_config = create_lane_encoder_config()
        policy.add_encoder(lane_config, freeze_existing=True)

        all_params = sum(1 for _ in policy.parameters())
        trainable_params = len(policy.get_trainable_params())

        # Should have fewer trainable params with frozen encoders
        assert trainable_params < all_params

    def test_export_onnx(self, policy):
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        # Export after file handle is closed
        policy.export_onnx(onnx_path)

        # Verify file exists
        assert Path(onnx_path).exists()

        # Try to load with ONNX (if available)
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
        except ImportError:
            pass  # ONNX not installed

        # Cleanup
        Path(onnx_path).unlink()


class TestTwoPhaseTraining:
    """Test two-phase training workflow"""

    def test_phase_b_to_c1_transition(self):
        """Simulate Phase B to Phase C-1 transition"""

        # Step 1: Create Phase B policy (242D)
        config_b = create_modular_policy_config_phase_b()
        policy_b = ModularDrivingPolicy(config_b)

        # Simulate training (just set some weights)
        with torch.no_grad():
            for param in policy_b.encoder.encoder_modules["ego"].parameters():
                param.fill_(1.0)

        # Step 2: Save Phase B checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save({
                'model_state_dict': policy_b.state_dict(),
                'avg_reward': 903.0,
            }, f.name)
            checkpoint_path = f.name

        # Step 3: Create Phase C-1 policy (254D) and load Phase B weights
        config_c1 = create_modular_policy_config_phase_b()
        policy_c1 = ModularDrivingPolicy(config_c1)

        # Load encoder weights
        loaded = policy_c1.load_encoder_weights(checkpoint_path)
        assert len(loaded) == 5

        # Step 4: Add lane encoder and freeze existing
        lane_config = create_lane_encoder_config()
        policy_c1.add_encoder(lane_config, freeze_existing=True)

        assert policy_c1.total_obs_dim == 254

        # Verify existing encoders are frozen
        for name in ["ego", "history", "agents", "route", "speed"]:
            assert policy_c1.encoder.encoder_modules[name].is_frozen

        # Verify lane encoder is trainable
        assert not policy_c1.encoder.encoder_modules["lane"].is_frozen

        # Verify ego weights were preserved
        for param in policy_c1.encoder.encoder_modules["ego"].parameters():
            assert torch.allclose(param, torch.ones_like(param))

        # Step 5: Simulate Phase 1 training (only trainable params)
        trainable_params = policy_c1.get_trainable_params()
        optimizer = torch.optim.Adam(trainable_params, lr=1.5e-4)

        obs = torch.randn(4, 254)
        loss = policy_c1(obs)['value'].mean()
        loss.backward()
        optimizer.step()

        # Step 6: Unfreeze all for Phase 2
        policy_c1.unfreeze_all_encoders()

        for name in policy_c1.encoder.encoder_modules:
            assert not policy_c1.encoder.encoder_modules[name].is_frozen

        # Cleanup
        Path(checkpoint_path).unlink()


class TestGradientFlow:
    """Test gradient flow with frozen encoders"""

    def test_frozen_encoder_no_gradients(self):
        config = create_modular_policy_config_phase_b()
        policy = ModularDrivingPolicy(config)

        # Freeze ego encoder
        policy.freeze_encoder("ego")

        # Forward + backward
        obs = torch.randn(4, 242)
        output = policy(obs)
        loss = output['value'].mean() + output['action'].mean()
        loss.backward()

        # Ego encoder should have no gradients
        for param in policy.encoder.encoder_modules["ego"].parameters():
            assert param.grad is None

        # Other encoders should have gradients
        for param in policy.encoder.encoder_modules["history"].parameters():
            assert param.grad is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
