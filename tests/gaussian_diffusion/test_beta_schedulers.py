import pytest
import torch

from diffusion_models.gaussian_diffusion.beta_schedulers import \
    BaseBetaScheduler, LinearBetaScheduler, CosineBetaScheduler


class TestBaseBetaScheduler:
    @pytest.mark.parametrize(
        "num_steps, enforce_zero_terminal",
        [(10, False)]
    )
    def test_scheduler_initialization_raises_error(self, num_steps, enforce_zero_terminal):
        with pytest.raises(NotImplementedError):
            BaseBetaScheduler(num_steps, enforce_zero_terminal, initialize=True)

    @pytest.mark.parametrize(
        "num_steps, enforce_zero_terminal",
        [(10, False)]
    )
    def test_scheduler_initialization_no_error(self, num_steps, enforce_zero_terminal):
        scheduler = BaseBetaScheduler(num_steps, enforce_zero_terminal, initialize=False)
        assert isinstance(scheduler.steps, int)
        assert scheduler.steps == num_steps
        assert scheduler.betas is None
        assert scheduler.alpha_bars is None

    def test_sample_betas_raises_error(self):
        scheduler = BaseBetaScheduler(10, False, initialize=False)
        with pytest.raises(NotImplementedError):
            scheduler.sample_betas()

    def test_compute_alpha_bar_raises_error(self):
        scheduler = BaseBetaScheduler(10, False, initialize=False)
        with pytest.raises(NotImplementedError):
            scheduler.compute_alpha_bar()

    @pytest.fixture
    def base_scheduler(self):
        return BaseBetaScheduler(steps=5, initialize=False)

    def test_initialization(self, base_scheduler):
        assert base_scheduler.steps == 5
        assert base_scheduler.betas is None
        assert base_scheduler.alpha_bars is None

    def test_to(self, base_scheduler):
        base_scheduler.betas = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        base_scheduler.alpha_bars = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        base_scheduler = base_scheduler.to('cpu')

        assert base_scheduler.betas.device == torch.device('cpu')
        assert base_scheduler.alpha_bars.device == torch.device('cpu')

    @pytest.mark.parametrize("steps, betas, alpha_bars", [
        (4, torch.tensor([0.2, 0.4, 0.6, 0.8]), torch.tensor([0.8, 0.6, 0.4, 0.2]))
    ])
    def test_from_tensors(self, base_scheduler, steps, betas, alpha_bars):
        scheduler = BaseBetaScheduler.from_tensors(steps, betas, alpha_bars)

        assert scheduler.steps == steps
        assert torch.equal(scheduler.betas, betas)
        assert torch.equal(scheduler.alpha_bars, alpha_bars)


class TestLinearBetaScheduler:

    def test_linear_beta_scheduler_init(self, mocker):
        mock_zero_terminal_snr = mocker.patch.object(
            LinearBetaScheduler, "enforce_zero_terminal_snr"
        )

        linear_beta_scheduler = LinearBetaScheduler(beta_start=0.0002, beta_end=0.03, steps=500, enforce_zero_terminal_snr=False)

        assert linear_beta_scheduler.beta_start == 0.0002
        assert linear_beta_scheduler.beta_end == 0.03
        assert linear_beta_scheduler.steps == 500

        # Test that zero_terminal_snr is not called when enforce_zero_terminal_snr is False
        mock_zero_terminal_snr.assert_not_called()

        linear_beta_scheduler = LinearBetaScheduler(beta_start=0.0002, beta_end=0.03, steps=500, enforce_zero_terminal_snr=True)

        # Test that zero_terminal_snr is called when enforce_zero_terminal_snr is True
        mock_zero_terminal_snr.assert_called_once()

    def test_sample_betas(self):
        linear_beta_scheduler = LinearBetaScheduler(beta_start=0.001, beta_end=0.01, steps=100)

        betas = linear_beta_scheduler.sample_betas()

        assert isinstance(betas, torch.Tensor)
        assert len(betas) == 100
        assert betas[0] == 0.001
        assert betas[-1] == 0.01

    def test_compute_alpha_bar(self):
        linear_beta_scheduler = LinearBetaScheduler(beta_start=0.0002, beta_end=0.03, steps=10)

        alpha_bar = linear_beta_scheduler.compute_alpha_bar()

        assert isinstance(alpha_bar, torch.Tensor)
        assert len(alpha_bar) == 10
        assert torch.isclose(alpha_bar[0], torch.tensor(1 - 0.0002))
        assert alpha_bar[-1] <= 1 - 0.03
        assert alpha_bar[0] > alpha_bar[-1]


class TestCosineBetaScheduler:

    def setup_method(self):
        self.offset = 0.008
        self.steps = 1000
        self.max_beta = 0.999
        self.cos_beta_scheduler = CosineBetaScheduler(
            self.offset, self.steps, self.max_beta
        )

    def test_f(self):
        t = torch.tensor([0, 1, 2], dtype=torch.float32)
        results = self.cos_beta_scheduler.f(t)

        assert results.shape == t.shape
        assert torch.allclose(
            results, torch.cos(
                (((t / self.steps) + self.offset) / (1 + self.offset)) * (
                      torch.pi / 2)
            ) ** 2
        )

    def test__compute_alpha_bar(self):
        results = self.cos_beta_scheduler._compute_alpha_bar()

        t = torch.linspace(0, self.steps, self.steps, dtype=torch.float32)
        expected_results = self.cos_beta_scheduler.f(
            t
        ) / self.cos_beta_scheduler.f(torch.tensor([0], dtype=torch.float32))

        assert results.shape == expected_results.shape
        assert torch.allclose(results, expected_results)

    def test__compute_betas(self):
        results = self.cos_beta_scheduler._compute_betas()

        alpha_bars = self.cos_beta_scheduler._compute_alpha_bar()
        expected_results = 1 - alpha_bars[1:] / alpha_bars[:-1]
        if self.max_beta:
            expected_results = torch.clip(expected_results, max=self.max_beta)

        assert results.shape == expected_results.shape
        assert torch.allclose(results, expected_results)

    def test_sample_betas(self):
        results = self.cos_beta_scheduler.sample_betas()

        betas = self.cos_beta_scheduler._compute_betas()

        assert torch.allclose(results, betas)

    def test_compute_alpha_bar(self):
        results = self.cos_beta_scheduler.compute_alpha_bar()

        alpha_bars = self.cos_beta_scheduler._compute_alpha_bar()

        assert torch.allclose(results, alpha_bars)
