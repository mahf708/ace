import dataclasses
import math
import warnings
from collections.abc import Callable
from typing import Literal

import torch

from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed.distributed import Distributed
from fme.core.models.conditional_sfno.sfnonet import (
    Context,
    ContextConfig,
    SFNONetConfig,
    get_lat_lon_sfnonet,
)


def _make_local_pos_embed_load_hook(
    param_name: str,
    img_shape: tuple[int, int],
    extra_leading_dims: int = 1,
) -> Callable:
    """Build a state-dict pre-hook that slices a legacy global-shape
    parameter to the local spatial extent expected by this rank.

    ``extra_leading_dims`` is the number of dims before the spatial (H, W) on
    the parameter (e.g. 1 for a (1, C, H, W) pos_embed, 2 for a
    (L, C, H, W) label_pos_embed).
    """

    def _hook(
        module: torch.nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        key = prefix + param_name
        if key not in state_dict:
            return
        tensor = state_dict[key]
        # Already local (saved by a sharded checkpoint at the same topology
        # or non-spatial single-rank): leave it alone.
        if tensor.shape[-2:] == tuple(getattr(module, param_name).shape[-2:]):
            return
        # Legacy global-shape: slice to local extent.
        if tensor.shape[-2:] != img_shape:
            # Shape unrecognized; let the standard load_state_dict error
            # path complain.
            return
        h_slice, w_slice = Distributed.get_instance().get_local_slices(img_shape)
        state_dict[key] = tensor[(..., h_slice, w_slice)].contiguous()
        warnings.warn(
            f"Loaded legacy global-shape '{key}' (shape {tuple(tensor.shape)}) "
            "and sliced it to the local spatial extent. Re-save the "
            "checkpoint to suppress this warning.",
            DeprecationWarning,
            stacklevel=2,
        )

    return _hook


def isotropic_noise(
    leading_shape: tuple[int, ...],
    lmax: int,  # length of the ℓ axis expected by isht (global)
    mmax: int,  # length of the m axis expected by isht (global)
    isht: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    dist = Distributed.get_instance()

    # --- draw independent N(0,1) parts --------------------------------------
    # Draw at full spectral extent on every rank; broadcast from spatial-rank
    # 0 so all spatial co-ranks see identical coefficients (otherwise the
    # same logical sample would see independent noise on each spatial rank).
    coeff_shape = (*leading_shape, lmax, mmax)
    real = torch.randn(coeff_shape, dtype=torch.float32, device=device)
    imag = torch.randn(coeff_shape, dtype=torch.float32, device=device)
    real = dist.broadcast_spatial(real)
    imag = dist.broadcast_spatial(imag)
    imag[..., :, 0] = 0.0  # m = 0 ⇒ purely real

    # m > 0: make Re and Im each N(0,½)  → |a_{ℓ m}|² has variance 1
    sqrt2 = math.sqrt(2.0)
    real[..., :, 1:] /= sqrt2
    imag[..., :, 1:] /= sqrt2

    # --- global scale that makes Var[T(θ,φ)] = 1 ---------------------------
    scale = math.sqrt(4.0 * math.pi) / lmax  # (Unsöld theorem ⇒ L = lmax)
    alm = (real + 1j * imag) * scale

    # --- for distributed iSHT, slice to local spectral extent --------------
    l_slice, m_slice = dist.get_local_slices((lmax, mmax))
    alm = alm[..., l_slice, m_slice]

    return isht(alm)


class NoiseConditionedModel(torch.nn.Module):
    """Wraps a context-based module with noise and optional label conditioning.

    Generates noise (gaussian by default, or isotropic via an inverse SHT)
    and optional positional embeddings (with label-position interaction),
    then calls the wrapped module with a fully populated Context.

    Args:
        conditional_model: An nn.Module with forward signature
            (x, context: Context).
        img_shape: Global spatial dimensions (lat, lon) of the input data.
        embed_dim_noise: Dimension of noise channels.
        embed_dim_pos: Dimension of learned positional embedding. 0 disables.
        embed_dim_labels: Dimension of label embeddings. 0 disables.
        inverse_sht: Optional inverse spherical harmonic transform callable.
            If provided, isotropic noise is generated via SHT; otherwise
            gaussian noise is used.
    """

    def __init__(
        self,
        conditional_model: torch.nn.Module,
        img_shape: tuple[int, int],
        embed_dim_noise: int = 256,
        embed_dim_pos: int = 0,
        embed_dim_labels: int = 0,
        inverse_sht: Callable[[torch.Tensor], torch.Tensor] | None = None,
        lmax: int = 0,
        mmax: int = 0,
    ):
        super().__init__()
        self.conditional_model = conditional_model
        self.embed_dim = embed_dim_noise
        self.img_shape = img_shape
        self._inverse_sht = inverse_sht
        self._lmax = lmax
        self._mmax = mmax
        self.label_pos_embed: torch.nn.Parameter | None = None
        # Compute local spatial extent so pos_embed parameters are sized to
        # what this rank actually uses (rather than full global + slice).
        h_slice, w_slice = Distributed.get_instance().get_local_slices(img_shape)
        # register pos embed if pos_embed_dim != 0
        if embed_dim_pos != 0:
            # Draw the global parameter and slice it to local so that the
            # final per-rank values match what a single-rank run would
            # produce at the same modes (preserves cross-topology
            # equivalence).
            global_pos = torch.zeros(1, embed_dim_pos, *img_shape)
            torch.nn.init.trunc_normal_(global_pos, std=0.02)
            self.pos_embed = torch.nn.Parameter(
                global_pos[..., h_slice, w_slice].clone().contiguous()
            )
            self.pos_embed._spatially_sharded = True  # type: ignore[attr-defined]
            self._register_load_state_dict_pre_hook(
                _make_local_pos_embed_load_hook("pos_embed", img_shape)
            )
            if embed_dim_labels > 0:
                global_label_pos = torch.zeros(
                    embed_dim_labels, embed_dim_pos, *img_shape
                )
                torch.nn.init.trunc_normal_(global_label_pos, std=0.02)
                self.label_pos_embed = torch.nn.Parameter(
                    global_label_pos[..., h_slice, w_slice].clone().contiguous()
                )
                self.label_pos_embed._spatially_sharded = True  # type: ignore[attr-defined]
                self._register_load_state_dict_pre_hook(
                    _make_local_pos_embed_load_hook("label_pos_embed", img_shape)
                )
        else:
            self.pos_embed = None

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x.reshape(-1, *x.shape[-3:])
        dist = Distributed.get_instance()
        if self._inverse_sht is not None:
            noise = isotropic_noise(
                (x.shape[0], self.embed_dim),
                self._lmax,
                self._mmax,
                self._inverse_sht,
                device=x.device,
            )
        else:
            # Draw at full global spatial extent and broadcast from spatial-
            # rank 0; otherwise spatial co-ranks would see independent noise
            # for the same logical sample. Sliced to local extent below.
            noise = torch.randn(
                [x.shape[0], self.embed_dim, *self.img_shape],
                device=x.device,
                dtype=x.dtype,
            )
            noise = dist.broadcast_spatial(noise)

        if self._inverse_sht is None:
            h_slice, w_slice = dist.get_local_slices(self.img_shape)
            noise = noise[..., h_slice, w_slice].contiguous()

        if self.pos_embed is not None:
            embedding_pos = self.pos_embed.repeat(noise.shape[0], 1, 1, 1)
            if self.label_pos_embed is not None and labels is not None:
                label_embedding_pos = torch.einsum(
                    "bl, lpxy -> bpxy", labels, self.label_pos_embed
                )
                embedding_pos = embedding_pos + label_embedding_pos
        else:
            embedding_pos = None

        return self.conditional_model(
            x,
            Context(
                embedding_scalar=None,
                embedding_pos=embedding_pos,
                labels=labels,
                noise=noise,
            ),
        )


# Backward-compatible alias
NoiseConditionedSFNO = NoiseConditionedModel


# this is based on the call signature of SphericalFourierNeuralOperatorNet at
# https://github.com/NVIDIA/modulus/blob/b8e27c5c4ebc409e53adaba9832138743ede2785/modulus/models/sfno/sfnonet.py#L292  # noqa: E501
@ModuleSelector.register("NoiseConditionedSFNO")
@dataclasses.dataclass
class NoiseConditionedSFNOBuilder(ModuleConfig):
    """
    Configuration for a noise-conditioned SFNO model.

    Noise is provided as conditioning input to conditional layer normalization.

    Attributes:
        spectral_transform: Unused, kept for backwards compatibility only.
        filter_type: Type of filter to use.
        operator_type: Unused, kept for backwards compatibility only.
            Must be "dhconv".
        residual_filter_factor: Factor by which to downsample the residual.
        embed_dim: Dimension of the embedding.
        noise_embed_dim: Dimension of the noise embedding.
        noise_type: Type of noise to use for conditioning.
        context_pos_embed_dim: Dimension of the position embedding to use
            for conditioning.
        global_layer_norm: Whether to reduce along the spatial domain when applying
            layer normalization.
        num_layers: Number of blocks (SFNO and MLP) in the model.
        use_mlp: Whether to use an MLP in the model.
        mlp_ratio: Ratio of the MLP hidden dimension
            to the embedding dimension.
        activation_function: Activation function to use.
        encoder_layers: Number of encoder layers in the model.
        pos_embed: Whether to use a position embedding.
        big_skip: Whether to use a big skip connection in the model.
        rank: Unused, kept for backwards compatibility only.
        factorization: Unused, kept for backwards compatibility only.
            Must be None.
        separable: Unused, kept for backwards compatibility only.
            Must be False.
        complex_network: Unused, kept for backwards compatibility only.
        complex_activation: Unused, kept for backwards compatibility only.
        spectral_layers: Unused, kept for backwards compatibility only.
        checkpointing: Whether to use checkpointing.
        data_grid: Grid type for spherical harmonic transforms.
        filter_residual: Whether to filter residual connections through a
            SHT round-trip. These will always be filtered if residual_filter_factor
            is not 1.
        filter_output: Whether to filter the output of the model through a
            SHT round-trip.
        local_blocks: List of block indices to use discrete-conditional
            convolution (DISCO) blocks, which apply local filters. See
            Ocampo et al. (2022)
            https://arxiv.org/abs/2209.13603 for more details.
        normalize_big_skip: Whether to normalize the big_skip connection.
        affine_norms: Whether to use element-wise affine parameters in the
            normalization layers.
        filter_num_groups: Number of groups to use in grouped convolutions
            for the spectral filter.
        lora_rank: Rank of the LoRA adaptations outside of spectral convolutions.
            0 (default) disables LoRA.
        lora_alpha: Strength of the LoRA adaptations outside of spectral convolutions.
            Defaults to lora_rank.
        spectral_lora_rank: Rank of the LoRA adaptations for spectral convolutions.
            0 (default) disables LoRA.
        spectral_lora_alpha: Strength of the LoRA adaptations for spectral convolutions.
            Defaults to spectral_lora_rank.
        filter_preserves_global_mean: If True, the spectral filter preserves
            the l=0 (global mean) spherical harmonic coefficient, so that
            global mean changes can only result from local operations
            (norms, MLPs, skip connections).
    """

    spectral_transform: Literal["sht"] = "sht"
    filter_type: Literal["linear", "makani-linear"] = "linear"
    operator_type: Literal["dhconv"] = "dhconv"
    residual_filter_factor: int = 1
    embed_dim: int = 256
    noise_embed_dim: int = 256
    context_pos_embed_dim: int = 0
    noise_type: Literal["isotropic", "gaussian"] = "gaussian"
    global_layer_norm: bool = False
    num_layers: int = 12
    use_mlp: bool = True
    mlp_ratio: float = 2.0
    activation_function: str = "gelu"
    encoder_layers: int = 1
    pos_embed: bool = True
    big_skip: bool = True
    rank: float = 1.0
    factorization: None = None
    separable: bool = False
    complex_network: bool = True
    complex_activation: str = "real"
    spectral_layers: int = 1
    checkpointing: int = 0
    # healpix not supported due to assumptions about number of spatial dims
    data_grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"
    filter_residual: bool = False
    filter_output: bool = False
    local_blocks: list[int] | None = None
    normalize_big_skip: bool = False
    affine_norms: bool = False
    filter_num_groups: int = 1
    lora_rank: int = 0
    lora_alpha: float | None = None
    spectral_lora_rank: int = 0
    spectral_lora_alpha: float | None = None
    filter_preserves_global_mean: bool = False

    def __post_init__(self):
        if self.context_pos_embed_dim > 0 and self.pos_embed:
            raise ValueError(
                "context_pos_embed_dim and pos_embed should not both be set"
            )
        if self.factorization is not None:
            raise ValueError("The 'factorization' parameter is no longer supported.")
        if self.separable:
            raise ValueError("The 'separable' parameter is no longer supported.")
        if self.operator_type != "dhconv":
            raise ValueError(
                "Only 'dhconv' operator_type is supported for "
                "NoiseConditionedSFNO models."
            )

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ):
        sfno_config = SFNONetConfig(
            embed_dim=self.embed_dim,
            filter_type=self.filter_type,
            global_layer_norm=self.global_layer_norm,
            num_layers=self.num_layers,
            use_mlp=self.use_mlp,
            mlp_ratio=self.mlp_ratio,
            activation_function=self.activation_function,
            encoder_layers=self.encoder_layers,
            pos_embed=self.pos_embed,
            big_skip=self.big_skip,
            checkpointing=self.checkpointing,
            filter_residual=self.filter_residual,
            filter_output=self.filter_output,
            local_blocks=self.local_blocks,
            normalize_big_skip=self.normalize_big_skip,
            affine_norms=self.affine_norms,
            filter_num_groups=self.filter_num_groups,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            spectral_lora_rank=self.spectral_lora_rank,
            spectral_lora_alpha=self.spectral_lora_alpha,
            filter_preserves_global_mean=self.filter_preserves_global_mean,
        )
        sfno_net = get_lat_lon_sfnonet(
            params=sfno_config,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=dataset_info.img_shape,
            data_grid=self.data_grid,
            context_config=ContextConfig(
                embed_dim_scalar=0,
                embed_dim_pos=self.context_pos_embed_dim,
                embed_dim_noise=self.noise_embed_dim,
                embed_dim_labels=len(dataset_info.all_labels),
            ),
        )
        if self.noise_type == "isotropic":
            inverse_sht = sfno_net.itrans_up
            lmax = inverse_sht.lmax
            mmax = inverse_sht.mmax
        else:
            inverse_sht = None
            lmax = 0
            mmax = 0
        return NoiseConditionedModel(
            sfno_net,
            embed_dim_noise=self.noise_embed_dim,
            embed_dim_pos=self.context_pos_embed_dim,
            embed_dim_labels=len(dataset_info.all_labels),
            img_shape=dataset_info.img_shape,
            inverse_sht=inverse_sht,
            lmax=lmax,
            mmax=mmax,
        )
